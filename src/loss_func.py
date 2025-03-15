import torch
import torch.nn as nn
import torch.nn.functional as F
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.ILP import ILP_solver
from torch import Tensor
from IPython.core.debugger import Tracer
    
    
class PMLoss(nn.Module):

    def __init__(self):
         super(PMLoss, self).__init__()
    
    
    
        
    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, possible_mat: Tensor, alpha: Tensor, beta: Tensor, lambda_value, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        batch_num = pred_dsmat.shape[0]

    
        pred_dsmat = torch.clamp(pred_dsmat, min=0.0, max=1.0)
        
        pred_dsmat1 = pred_dsmat
        
        alpha=alpha
        beta=beta
        
        
        possible_mat = possible_mat
        ali_perm = gt_perm+possible_mat
        ali_perm[ali_perm > 1.0] = 1.0 
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
      
        
        loss = torch.tensor(0.).to(pred_dsmat.device)
        loss2 = torch.tensor(0.).to(pred_dsmat.device)
        loss3 = torch.tensor(0.).to(pred_dsmat.device)
        
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            
            
            
            gt_row = torch.sum(gt_perm[b, :src_ns[b], :tgt_ns[b]],dim=1)
            gt_column= torch.sum(gt_perm[b, :src_ns[b], :tgt_ns[b]],dim=0)
            
            alpha_sel = alpha[b,:src_ns[b]]
            beta_sel = beta[b,:tgt_ns[b]]
            
            
            alpha_sel = torch.mul(gt_row,alpha_sel)
            beta_sel = torch.mul(gt_column,beta_sel)
            
            
            loss += F.binary_cross_entropy(
                pred_dsmat[b, :src_ns[b], :tgt_ns[b]],
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                reduction='sum')+lambda_value*(F.mse_loss(alpha_sel, gt_row,reduction='sum')+F.mse_loss(beta_sel, gt_column,reduction='sum'))
            
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
      
        
        
        return (loss) / n_sum
        
    
    
    
