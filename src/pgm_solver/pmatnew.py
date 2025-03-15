import torch
from torch.multiprocessing import Pool
from IPython.core.debugger import Tracer
import torch.nn as nn



def check_nan_inf(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return True
    else:
        return False    

class PMATA(nn.Module):
    
    
    def __init__(self,wm1 : torch.Tensor ,wm2: torch.Tensor,wm3: torch.Tensor):
        super(PMATA, self).__init__()
        self.wm1 = wm1
        self.wm2 =wm2
        self.wm3 =wm3
    
    def forward(self,s: torch.Tensor,os: torch.Tensor, rho, n1: torch.Tensor=None, n2: torch.Tensor=None,  nproc: int=1) -> torch.Tensor:
       
        
        
       
        
        
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood: {}'.format(s.shape))
            
        if len(os.shape) == 2:
            os = os.unsqueeze(0)
            #matrix_input = True
        elif len(os.shape) == 3:
            pass
            #matrix_input = False
        else:
            raise ValueError('input data shape not understood: {}'.format(os.shape))

        s1 = s
        os1=os
        n1d = n1
        n2d = n2

       

        device = s.device
        

        batch_num = s.shape[0]

        perm_mat = s
        os_mat=os
        output_list = []
        perm_mat_list = []
       
        alpha_list = []
        beta_list = []
        

        if n1 is not None:
            n1 = n1
        else:
            n1 = [None] * batch_num
        if n2 is not None:
            n2 = n2
        else:
            n2 = [None] * batch_num

         
        
        for b in range(batch_num):
            # Calculate the tensor using _pmatkernelnew() and append it to the list
            
            perm_mat_o,alpha_o,beta_o=self._pmatkernelnew(perm_mat[b],os_mat[b], rho, n1[b], n2[b])
            perm_mat_list.append(perm_mat_o)
            alpha_list.append(alpha_o)
            beta_list.append(beta_o)
            

        # Concatenate tensors in the list along a new dimension
        perm_mat = torch.stack(perm_mat_list)
        alpha = torch.stack(alpha_list)
        beta = torch.stack(beta_list)
        
        

        if matrix_input:
            perm_mat.squeeze_(0)
            alpha.squeeze_(0)
            beta.squeeze_(0)

        output_list.append(perm_mat)
        output_list.append(alpha)
        output_list.append(beta)
        
        return output_list
    
    def _pmatkernelnew(self, s: torch.Tensor, os: torch.Tensor, rho, n1=None, n2=None):

        smat=s
        osmat=os

        if n1 is None:
            n1 = s.shape[0]
        if n2 is None:
            n2 = s.shape[1]



        n1d = n1
        n2d = n2

        rho = rho
        
       


        
        s_sliced = s[:n1, :n2]
        os_sliced = os[:n1, :n2]
        c_mat = 1 - s_sliced

        os_sliced=torch.clamp(os_sliced, min=0.0)
        alpha_6,_=torch.max(os_sliced,dim=1)
        beta_6,_=torch.max(os_sliced,dim=0)
        wm1 = torch.clamp(self.wm1, min=0.0)
        alpha_7 = (torch.sigmoid(wm1*alpha_6)-0.5)*2
        beta_7 = (torch.sigmoid(wm1*beta_6)-0.5)*2
        alpha = alpha_7
        beta = beta_7
        
        w1 = alpha
        w2 = beta
       

      
        wm = 1000

        n = max(n1,n2)
        possible_mat = 0

        if (n1 < n):
            diff = n - n1
            
          
           
            
            ws = torch.ones(diff) * 1.1
            ws = ws.to('cuda')


         
            w1 = torch.cat((w1, ws), 0)
           
            
            
            w1 = w1.reshape(-1, 1)
            w2 = w2.reshape(1, -1)

           


            w1 = w1.expand(n, n)
            w2 = w2.expand(n, n)

            c_base = ((w1 + w2) * rho)




            c_base_sliced = c_base[:n1, :n2]
            possible_mat_t = torch.where(c_mat <= c_base_sliced, 1, 0)

            s_size_mat_row_difference = s.shape[0] - n1
            s_size_mat_column_difference = s.shape[1] - n2
            possible_mat = torch.nn.functional.pad(possible_mat_t, (0,s_size_mat_column_difference, 0, s_size_mat_row_difference ), 'constant', 0)
            
              
           
           
            
            wzr = torch.zeros(s_size_mat_row_difference).to('cuda')
            wzc = torch.zeros(s_size_mat_column_difference).to('cuda') 
            
            alpha = torch.cat((alpha, wzr), 0)
            beta = torch.cat((beta, wzc), 0)

        elif (n1 >= n):
            diff = n - n2
            ws = torch.ones(diff) * 1.1
            ws = ws.to('cuda')

            w2 = torch.cat((w2, ws), 0)
            
            
            w1 = w1.reshape(-1, 1)
            w2 = w2.reshape(1, -1)

            w1 = w1.expand(n, n)
            w2 = w2.expand(n, n)

            c_base = ((w1 + w2) * rho)

            c_base_sliced = c_base[:n1, :n2]
            possible_mat_t = torch.where(c_mat <= c_base_sliced, 1, 0)

            s_size_mat_row_difference = s.shape[0] - n1
            s_size_mat_column_difference = s.shape[1] - n2
            possible_mat = torch.nn.functional.pad(possible_mat_t, (0,s_size_mat_column_difference, 0, s_size_mat_row_difference ), 'constant', 0)

       
            wzr = torch.zeros(s_size_mat_row_difference).to('cuda')
            wzc = torch.zeros(s_size_mat_column_difference).to('cuda') 
            
            alpha = torch.cat((alpha, wzr), 0)
            beta = torch.cat((beta, wzc), 0)


        return possible_mat,alpha,beta
        
    
   
