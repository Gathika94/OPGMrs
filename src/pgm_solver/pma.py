import torch
import scipy.optimize as opt
import numpy as np
from multiprocessing import Pool
from torch import Tensor
from IPython.core.debugger import Tracer


def pma(s: Tensor, os: Tensor, wm1: torch.Tensor, wm2: torch.Tensor, wm3: torch.Tensor, rho, n1: Tensor=None, n2: Tensor=None, nproc: int=1) -> Tensor:
    
    
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))
        
    
    if len(os.shape) == 2:
        os = os.unsqueeze(0)
    elif len(os.shape) == 3:
        pass
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))

    s1 = s
    os1=os
    n1d = n1
    n2d = n2
    
   
    
    device = s.device
    
    
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach()
    os_mat= os.cpu().detach()
    
    
    
    wm1 = wm1.cpu().detach()
    wm2 = wm2.cpu().detach()
    wm3 = wm3.cpu().detach()
    
    if n1 is not None:
        n1 = n1.cpu().detach()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.cpu().detach()
    else:
        n2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_pma_kernel, zip(perm_mat, os_mat, wm1, wm2, wm3, rho, n1, n2))
            perm_mat = np.stack(mapresult.get())
            
    else:
        perm_mat = np.stack([_pma_kernel(perm_mat[b], os_mat[b], wm1, wm2, wm3, rho, n1[b], n2[b]) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)
    wm1 = wm1.to(device)
    wm2 = wm2.to(device)
    wm3 = wm3.to(device)
    
    
    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat

def _pma_kernel(s: torch.Tensor, os: torch.Tensor,  wm1: torch.Tensor, wm2: torch.Tensor, wm3: torch.Tensor, rho, n1=None, n2=None):
       
        
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    
    n1d = n1
    n2d = n2
        
    rho= rho
    
   
  
    s_mat = s
    s_sliced = s[:n1, :n2]
    os_sliced = os[:n1, :n2]
    c_mat = 1 - s_sliced
    
    os_sliced = torch.clamp(os_sliced, min=0.0)
    wm1 = torch.clamp(wm1, min=0.0)
        
    alpha_6,_=torch.max(os_sliced,dim=1)
    beta_6,_=torch.max(os_sliced,dim=0)
    
 
        
    alpha_7 = (torch.sigmoid(wm1*alpha_6)-0.5)*2
    beta_7 = (torch.sigmoid(wm1*beta_6)-0.5)*2
    
    alpha = alpha_7
    beta = beta_7
        
    
    
    w1 = alpha
    w2 = beta
    
 
    
    

    wm=10000
    
    n = max(n1,n2)
    perm_mat = 0
    
    if (n1 < n):
        diff=n-n1
        ws = torch.ones(diff)*1.1 
        w1 = torch.cat((w1,ws),0)
        w1 = w1.reshape(-1,1)
        w2 = w2.reshape(1, -1)
        
        w1 = w1.expand(n,n)
        w2 = w2.expand(n,n)
       
        
        c_base = ((w1+w2)*rho).numpy()
        
        c_base_sliced = c_base[:n1, :n2]
        c_mat = c_mat.numpy()
        
        possible_mat = np.where(c_mat <= c_base_sliced,1,0)
        
        c_padded = np.pad(c_mat,((0,diff),(0,0)),'constant', constant_values=wm)
        c_modified = np.where(c_padded <= c_base, c_padded, c_base)
        row, col = opt.linear_sum_assignment(c_modified)
        perm_mat_d1 = np.zeros_like(c_modified)
        perm_mat_d1[row, col] = 1
        
        perm_mat_t = perm_mat_d1[:n1, :n2]
        perm_mat_t1 = np.multiply(perm_mat_t,possible_mat)
       
        s_size_mat_row_difference = s.shape[0] - n1
        s_size_mat_column_difference = s.shape[1] - n2
        perm_mat = np.pad(perm_mat_t1,((0,s_size_mat_row_difference),(0,s_size_mat_column_difference)),'constant')
        
       
        
    elif (n2 <= n):
        diff=n-n2
        ws = torch.ones(diff)*1.1
        w2 = torch.cat((w2,ws),0)  
        w1 = w1.reshape(-1,1)
        w2 = w2.reshape(1, -1)
        
        w1 = w1.expand(n,n)
        w2 = w2.expand(n,n)
        
        c_base = ((w1+w2)*rho).numpy()
        
        c_base_sliced = c_base[:n1, :n2]
        c_mat = c_mat.numpy()
        
        possible_mat = np.where(c_mat <= c_base_sliced,1,0)
        
        c_padded = np.pad(c_mat,((0,0),(0,diff)),'constant', constant_values=wm)
        
        c_modified = np.where(c_padded <= c_base, c_padded, c_base)
        row, col = opt.linear_sum_assignment(c_modified)
        
       
        perm_mat_d1 = np.zeros_like(c_modified)
        perm_mat_d1[row, col] = 1
        
        perm_mat_t = perm_mat_d1[:n1, :n2]
        perm_mat_t1 = np.multiply(perm_mat_t,possible_mat)
        
        s_size_mat_row_difference = s.shape[0] - n1
        s_size_mat_column_difference = s.shape[1] - n2
        perm_mat = np.pad(perm_mat_t1,((0,s_size_mat_row_difference),(0,s_size_mat_column_difference)),'constant')
    
   
    return perm_mat


