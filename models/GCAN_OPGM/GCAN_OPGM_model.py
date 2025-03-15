import itertools

from models.GCAN_OPGM.positional_encoding_layer import positional_encoding_layer
from models.GCAN_OPGM.GCA_module import GCA_module
from models.GCAN_OPGM.adaptive_classifier import AdaptiveClassifier
from src.feature_align import feature_align
from src.utils.pad_tensor import pad_tensor, pad_tensor_varied
from src.lap_solvers.ILP import ILP_solver
from src.lap_solvers.sinkhorn import Sinkhorn as Sinkhorn_varied
from torch_geometric import utils as geometric_util
from scipy.linalg import block_diag
import numpy as np
from src.pgm_solver.pma import pma
from src.pgm_solver.pmatnew import PMATA

from src.utils.config import cfg

from src.backbone_gcan import *
from IPython.core.debugger import Tracer


CNN = eval(cfg.BACKBONE)

def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

def pair_split(node_features, ns):
    batch_feature_src = []
    batch_feature_tgt = []
    partitions = []
    idx = 0
    for i in range(len(ns)):
        partitions.append(node_features[idx:idx + ns[i], :])
        idx = idx + ns[i]
    for i in range(0, len(ns), 2):
        batch_feature_src.append(partitions[i])
        batch_feature_tgt.append(partitions[i + 1])
    return batch_feature_src,  batch_feature_tgt

def get_graph_feature(batch_graphs_src, ns_src, batch_graphs_tgt, ns_tgt):
    
    adjacency_matrixs_list = []
    adjacency_src_list = []
    adjacency_tgt_list = []
    for idx in range(len(batch_graphs_src)):
        
        adjacency_src = geometric_util.to_dense_adj(
            batch_graphs_src[idx].edge_index, max_num_nodes=ns_src[idx]).squeeze().cpu()
        #degrees_src = torch.sum(adjacency_src, dim=1)
        adjacency_src_list.append(np.array(adjacency_src))
        adjacency_matrixs_list.append(np.array(adjacency_src))
        
        
        adjacency_tgt = geometric_util.to_dense_adj(
            batch_graphs_tgt[idx].edge_index, max_num_nodes=ns_tgt[idx]).squeeze().cpu()
        #degrees_tgt = torch.sum(adjacency_tgt, dim=1)
        adjacency_tgt_list.append(np.array(adjacency_tgt))
        adjacency_matrixs_list.append(np.array(adjacency_tgt))
        
        
        
    
    adjacency_matrixs = block_diag(*adjacency_matrixs_list).astype('float32')
    
    
    
    max_size_dim0_src = max(array.shape[0] for array in adjacency_src_list) #improve this using ns 
    max_size_dim1_src = max(array.shape[1] for array in adjacency_src_list) #this should be same as above

   
    padded_tensor_list_src = []
    for array in adjacency_src_list:
        padded_array = torch.nn.functional.pad(torch.tensor(array), (0, max_size_dim1_src - array.shape[1], 0, max_size_dim0_src - array.shape[0]), value=0)
        padded_tensor_list_src.append(padded_array)

 
    adjacency_tensor_src = torch.stack(padded_tensor_list_src)
    
    
   
    max_size_dim0_tgt = max(array.shape[0] for array in adjacency_tgt_list)
    max_size_dim1_tgt = max(array.shape[1] for array in adjacency_tgt_list)

    
    padded_tensor_list_tgt = []
    for array in adjacency_tgt_list:
        padded_array = torch.nn.functional.pad(torch.tensor(array), (0, max_size_dim1_tgt - array.shape[1], 0, max_size_dim0_tgt - array.shape[0]), value=0)
        padded_tensor_list_tgt.append(padded_array)

   

    adjacency_tensor_tgt = torch.stack(padded_tensor_list_tgt)


    
    
    
    
    return torch.tensor(adjacency_matrixs).cuda(),adjacency_tensor_src.cuda(),adjacency_tensor_tgt.cuda()


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.positional_encoding = positional_encoding_layer(input_node_dim=cfg.GCAN.FEATURE_CHANNEL * 2)
        self.global_state_dim = cfg.GCAN.FEATURE_CHANNEL * 2
        cross_parameters = [self.global_state_dim, self.positional_encoding.num_node_features]
        self_parameters = [cfg.GCAN.NODE_HIDDEN_SIZE[-1]*2, int(cfg.GCAN.NODE_HIDDEN_SIZE[-1]/4), 4]
        self.GCA_module1 = GCA_module(cross_parameters,self_parameters)
        self.GCA_module2 = GCA_module(cross_parameters, self_parameters)

        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.GCAN.SK_TAU
        self.rho = cfg.OPGM.RHO
       
        self.sinkhorn = Sinkhorn_varied(max_iter=cfg.GCAN.SK_ITER_NUM, tau=self.tau, epsilon=cfg.GCAN.SK_EPSILON)
        
        
        #self.wm1 = nn.Parameter(torch.full((1,), 0.5))
        self.wm1 = nn.Parameter(torch.tensor(1.0),requires_grad=True) #changed
        self.wm2 = nn.Parameter(torch.tensor(0.8), requires_grad=True)
        self.wm3 = nn.Parameter(torch.tensor(0.8), requires_grad=True)
        self.acf = AdaptiveClassifier()
        self.pmata = PMATA(wm1=self.wm1,wm2 = self.wm2,wm3=self.wm3)
    
        self.count=0

    def forward(
        self,
        data_dict,
    ):
        
       
        
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        num_graphs = len(images)

        global_avg_list = []
        global_max_list = []
        orig_graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_avg_list.append(self.final_layers_avg(edges).reshape((nodes.shape[0], -1)))
            global_max_list.append(self.final_layers_max(edges).reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = torch.cat((U, F), dim=1)
            graph.x = node_features

            graph = self.positional_encoding(graph)
            orig_graph = graph.to_data_list()
            orig_graph_list.append(orig_graph)

        ns_src, ns_tgt = n_points
        P_src, P_tgt = points
        global_avg_src, global_avg_tgt = global_avg_list
        global_max_src, global_max_tgt = global_max_list
        batch_graphs_src, batch_graphs_tgt = orig_graph_list
        cross_attention_list = []
    
        global_max_weights = torch.cat([global_max_src, global_max_tgt], axis=-1)
        global_max_weights = normalize_over_channels(global_max_weights)
        global_avg_weights = torch.cat([global_avg_src, global_avg_tgt], axis=-1)
        global_avg_weights = normalize_over_channels(global_avg_weights)
     
        batch_feature_src = [item.x for item in batch_graphs_src]
        
        batch_feature_tgt = [item.x for item in batch_graphs_tgt]
        
        adjacency_matrixs, adjacency_src_list, adjacency_tgt_list  = get_graph_feature(batch_graphs_src, ns_src, batch_graphs_tgt, ns_tgt)
        
        
        cross_attention, node_features, ns = self.GCA_module1(batch_feature_src, batch_feature_tgt, global_avg_weights, global_max_weights, ns_src, ns_tgt,adjacency_matrixs)
        
        
        
        
        cross_attention_list = cross_attention_list + cross_attention
        batch_feature_src, batch_feature_tgt = pair_split(node_features, ns)
        cross_attention, node_features, ns = self.GCA_module2(batch_feature_src, batch_feature_tgt, global_avg_weights,
                                                          global_max_weights, ns_src, ns_tgt, adjacency_matrixs)
        cross_attention_list = [ori + 0.1*new for ori, new in zip(cross_attention_list, cross_attention)]

        s_list, os_list, mgm_s_list, x_list, mgm_x_list, alpha_list, beta_list, indices = [], [], [], [], [],[],[],[]

        for idx1, idx2 in lexico_iter(range(num_graphs)):
            if True:
                Kp = torch.stack(pad_tensor_varied(cross_attention_list,dummy=0), dim=0)
            else:
                Kp = torch.stack(pad_tensor(cross_attention_list), dim=0)

            s = Kp
            os = s
         
            
            if self.training:
                if True:
                    #print(s)
                   
                    
                    ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                    output_list = self.pmata(ss, os, self.rho, n_points[idx1], n_points[idx2])
                    pos_mat= output_list[0] 
                    alpha = output_list[1]
                    beta = output_list[2]
                  
                    
                else:
                    ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                   
                    output_list = self.pmata(ss, os,self.rho, n_points[idx1], n_points[idx2])
                    pos_mat= output_list[0] 
                    alpha = output_list[1]
                    beta = output_list[2]
                    
                s_list.append(ss)
                os_list.append(os)
                x_list.append(pos_mat)
                alpha_list.append(alpha)
                beta_list.append(beta)
                indices.append((idx1, idx2))
            else:
                if(self.count==0):
                    print(self.wm1.item())
                    print(self.wm2.item())
                    self.count=1
                    
                if True:
                    ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                   
                    alpha,beta = self.acf(os)
                    pred_mat = pma(ss, os,self.wm1, self.wm2, self.wm3, self.rho, n_points[idx1], n_points[idx2])
                   
                else:
                    ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                    
                    alpha,beta = self.acf(os)
                    pred_mat = pma(ss, os,self.wm1, self.wm2, self.wm3, self.rho, n_points[idx1], n_points[idx2])
                   
                
                    
                s_list.append(ss)
                os_list.append(os)
                x_list.append(pred_mat)
                alpha_list.append(pred_mat)
                beta_list.append(pred_mat)
                indices.append((idx1, idx2))
        
        
        
        
        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({
                'ds_mat': s_list[0],
                'os_mat': os_list[0],
                'perm_mat': x_list[0],
                'alpha_f':alpha_list[0],
                'beta_f': beta_list[0]
            })

        return data_dict
