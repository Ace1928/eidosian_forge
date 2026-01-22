import torch
import torch.nn.functional as F
def dropout_eval(x):
    return F.dropout(x, p=0.5, training=False, inplace=inplace)