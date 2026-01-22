import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
from torch.fx import GraphModule
from torch.fx.graph import Node
from .utils import (
from .ns_types import (
from typing import List, Optional, Dict, Callable
def get_op_to_type_to_weight_extraction_fn() -> Dict[str, Dict[Callable, Callable]]:
    op_to_type_to_weight_extraction_fn: Dict[str, Dict[Callable, Callable]] = {'call_module': {nn.Conv1d: mod_weight_detach, nni.ConvReLU1d: mod_0_weight_detach, nnq.Conv1d: mod_weight_bias_0, nnqat.Conv1d: mod_weight_detach, nniqat.ConvBn1d: mod_weight_detach, nniqat.ConvBnReLU1d: mod_weight_detach, nniqat.ConvReLU1d: mod_weight_detach, nniq.ConvReLU1d: mod_weight_bias_0, nn.Conv2d: mod_weight_detach, nni.ConvReLU2d: mod_0_weight_detach, nnq.Conv2d: mod_weight_bias_0, nnqat.Conv2d: mod_weight_detach, nniqat.ConvBn2d: mod_weight_detach, nniqat.ConvBnReLU2d: mod_weight_detach, nniqat.ConvReLU2d: mod_weight_detach, nniq.ConvReLU2d: mod_weight_bias_0, nn.Conv3d: mod_weight_detach, nni.ConvReLU3d: mod_0_weight_detach, nnq.Conv3d: mod_weight_bias_0, nnqat.Conv3d: mod_weight_detach, nniqat.ConvBn3d: mod_weight_detach, nniqat.ConvBnReLU3d: mod_weight_detach, nniqat.ConvReLU3d: mod_weight_detach, nniq.ConvReLU3d: mod_weight_bias_0, nn.Linear: mod_weight_detach, nnq.Linear: mod_weight_bias_0, nni.LinearReLU: mod_0_weight_detach, nniq.LinearReLU: mod_weight_bias_0, nnqat.Linear: mod_weight_detach, nnqd.Linear: mod_weight_bias_0, nniqat.LinearReLU: mod_weight_detach, nniqat.LinearBn1d: mod_weight_detach, nn.modules.linear.NonDynamicallyQuantizableLinear: mod_weight_detach, nn.LSTM: get_lstm_weight, nnqd.LSTM: get_qlstm_weight}, 'call_function': {F.conv1d: get_conv_fun_weight, F.conv2d: get_conv_fun_weight, F.conv3d: get_conv_fun_weight, toq.conv1d: get_qconv_fun_weight, toq.conv2d: get_qconv_fun_weight, toq.conv3d: get_qconv_fun_weight, toq.conv1d_relu: get_qconv_fun_weight, toq.conv2d_relu: get_qconv_fun_weight, toq.conv3d_relu: get_qconv_fun_weight, F.linear: get_linear_fun_weight, toq.linear: get_qlinear_fun_weight, toq.linear_relu: get_qlinear_fun_weight}}
    return op_to_type_to_weight_extraction_fn