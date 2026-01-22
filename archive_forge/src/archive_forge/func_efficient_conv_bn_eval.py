import torch
import torch.nn as nn
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch.func import functional_call
from ..pattern_matcher import CallModuleVarArgs, Match, register_graph_pattern
from .pre_grad import efficient_conv_bn_eval_pass
def efficient_conv_bn_eval(bn: nn.modules.batchnorm._BatchNorm, conv: nn.modules.conv._ConvNd, x: torch.Tensor):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Tune-Mode ConvBN Blocks For Efficient Transfer Learning"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for **training** as well, but only if one sets `bn.training=False`. It
     reduces memory footprint and computation cost, at the cost of slightly
     reduced numerical stability.
    Args:
        bn (nn.modules.batchnorm._BatchNorm): a BatchNorm module.
        conv (nn.modules.conv._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """
    assert bn.running_var is not None
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)
    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)
    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)
    target_shape = [-1] + [1] * (conv.weight.ndim - 1)
    if isinstance(conv, nn.modules.conv._ConvTransposeNd):
        target_shape[:2] = [target_shape[1], target_shape[0]]
    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape(target_shape)
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (bias_on_the_fly - bn.running_mean)
    input = x
    params = {'weight': weight_on_the_fly, 'bias': bias_on_the_fly}
    output = functional_call(conv, params, input)
    return output