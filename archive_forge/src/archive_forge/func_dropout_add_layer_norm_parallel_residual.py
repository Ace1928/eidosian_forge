import dropout_layer_norm
import torch
from torch.nn import init
def dropout_add_layer_norm_parallel_residual(x0, x1, residual, weight0, bias0, weight1, bias1, dropout_p, epsilon, prenorm=False, residual_in_fp32=False, return_dropout_mask=False):
    """residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    """
    return DropoutAddLayerNormParallelResidualFn.apply(x0, x1, residual, weight0, bias0, weight1, bias1, dropout_p, epsilon, residual_in_fp32, prenorm, False, return_dropout_mask)