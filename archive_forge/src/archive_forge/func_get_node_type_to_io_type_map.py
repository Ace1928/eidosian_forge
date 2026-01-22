import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.ao.nn.qat.dynamic as nnqatd
from torch.ao.quantization.backend_config import get_native_backend_config
import torch.ao.quantization.fx._lower_to_native_backend as \
import torch.ao.quantization.quantization_mappings as quantization_mappings
from .ns_types import NSNodeTargetType
from typing import Callable, Dict, List, Optional, Set, Tuple
def get_node_type_to_io_type_map() -> Dict[str, Set[NSNodeTargetType]]:
    FUNS_IO_TYPE_FP32: Set[NSNodeTargetType] = {F.linear, F.conv1d, F.conv2d, F.conv3d, torch.cat, F.elu, F.hardswish, F.instance_norm, F.layer_norm, F.leaky_relu, F.dropout, F.silu, F.mish, operator.add, torch.add, operator.mul, torch.mul, torch.sum, F.prelu}
    FUNS_IO_TYPE_FP16: Set[NSNodeTargetType] = set()
    FUNS_IO_TYPE_INT8: Set[NSNodeTargetType] = {toq.linear, toq.linear_relu, toq.conv1d, toq.conv1d_relu, toq.conv2d, toq.conv2d_relu, toq.conv3d, toq.conv3d_relu, toq.cat, toq.elu, toq.hardswish, toq.instance_norm, toq.layer_norm, toq.leaky_relu, toq.dropout, toq.prelu}
    FUNS_IO_TYPE_FP32_OR_INT8: Set[NSNodeTargetType] = {F.relu, F.tanh, torch.tanh, F.sigmoid, torch.sigmoid, F.hardsigmoid, operator.floordiv, torch.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d, F.dropout, F.hardtanh, F.hardtanh_, F.interpolate, F.max_pool1d, F.max_pool2d, F.max_pool3d, F.relu6, F.pixel_shuffle, F.pixel_unshuffle, torch.avg_pool1d, torch._C._nn.avg_pool2d, torch._C._nn.avg_pool3d, torch.cat, torch.chunk, torch.clamp, torch.flatten, torch.transpose, torch.max, torch.mean, torch.min, torch.narrow, torch.repeat_interleave, torch.sort, torch.squeeze, torch.stack, torch.unsqueeze, operator.add}
    MODS_IO_TYPE_FP32: Set[NSNodeTargetType] = {nn.Linear, nnqat.Linear, nnqatd.Linear, nnqd.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d, nnqat.Embedding, nnqat.EmbeddingBag, nn.LSTM, nnqd.LSTM, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.ELU, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm, nn.Hardswish, nn.LeakyReLU, nn.ReLU6, nn.SiLU, nn.Mish, nn.Softmax, nn.PReLU, nni.BNReLU2d, nni.BNReLU3d, nni.ConvReLU1d, nni.ConvReLU2d, nni.ConvReLU3d, nni.LinearReLU, nni.LinearBn1d, nni.ConvBn1d, nni.ConvBn2d, nni.ConvBn3d, nniqat.ConvBn1d, nniqat.ConvBn2d, nniqat.ConvBn3d, nniqat.ConvBnReLU1d, nniqat.ConvBnReLU2d, nniqat.ConvBnReLU3d, nniqat.ConvReLU1d, nniqat.ConvReLU2d, nniqat.ConvReLU3d, nniqat.LinearReLU, nniqat.LinearBn1d, nniqd.LinearReLU, nni.LinearLeakyReLU, nni.LinearTanh, nni.ConvAdd2d, nni.ConvAddReLU2d}
    MODS_IO_TYPE_INT8: Set[NSNodeTargetType] = {nnq.Linear, nnq.Conv1d, nnq.Conv2d, nnq.Conv3d, nnq.BatchNorm2d, nnq.BatchNorm3d, nnq.Dropout, nnq.ConvTranspose1d, nnq.ConvTranspose2d, nnq.ELU, nnq.InstanceNorm1d, nnq.InstanceNorm2d, nnq.InstanceNorm3d, nnq.LayerNorm, nnq.Hardswish, nnq.LeakyReLU, nnq.Embedding, nnq.EmbeddingBag, nnq.Dropout, nnq.Softmax, nnq.PReLU, nniq.BNReLU2d, nniq.BNReLU3d, nniq.ConvReLU1d, nniq.ConvReLU2d, nniq.ConvReLU3d, nniq.LinearReLU, nniq.LinearLeakyReLU, nniq.LinearTanh, nniq.ConvAdd2d, nniq.ConvAddReLU2d}
    MODS_IO_TYPE_FP32_OR_INT8: Set[NSNodeTargetType] = {nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Hardsigmoid, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.Dropout, nn.Hardtanh, nn.Identity, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.PixelShuffle, nn.PixelUnshuffle, nn.ReLU6}
    METHS_IO_TYPE_FP32_OR_INT8: Set[NSNodeTargetType] = {'sigmoid_', 'sigmoid', 'tanh_', 'tanh', 'hardsigmoid_', 'hardsigmoid', 'relu_', 'relu'}
    return {'funs_io_type_fp32': FUNS_IO_TYPE_FP32, 'funs_io_type_fp16': FUNS_IO_TYPE_FP16, 'funs_io_type_int8': FUNS_IO_TYPE_INT8, 'funs_io_type_fp32_or_int8': FUNS_IO_TYPE_FP32_OR_INT8, 'mods_io_type_fp32': MODS_IO_TYPE_FP32, 'mods_io_type_int8': MODS_IO_TYPE_INT8, 'mods_io_type_fp32_or_int8': MODS_IO_TYPE_FP32_OR_INT8, 'meths_io_type_fp32_or_int8': METHS_IO_TYPE_FP32_OR_INT8}