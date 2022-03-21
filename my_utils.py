import os
import random
import numpy as np

import torch
import torch.nn as nn


def set_seed(n_gpu, seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)     
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()