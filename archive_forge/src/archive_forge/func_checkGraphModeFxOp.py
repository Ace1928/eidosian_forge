import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.nn.intrinsic import _FusedModule
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM
from torch.ao.quantization import (
from torch.ao.quantization import QuantWrapper, QuantStub, DeQuantStub, \
from torch.ao.quantization.quantization_mappings import (
from torch.testing._internal.common_quantized import (
from torch.jit.mobile import _load_for_lite_interpreter
import copy
import io
import functools
import time
import os
import unittest
import numpy as np
from torch.testing import FileCheck
from typing import Callable, Tuple, Dict, Any, Union, Type, Optional
import torch._dynamo as torchdynamo
def checkGraphModeFxOp(self, model, inputs, quant_type, expected_node=None, expected_node_occurrence=None, expected_node_list=None, is_reference=False, print_debug_info=False, custom_qconfig_dict=None, prepare_expected_node=None, prepare_expected_node_occurrence=None, prepare_expected_node_list=None, prepare_custom_config=None, backend_config=None):
    """ Quantizes model with graph mode quantization on fx and check if the
                quantized model contains the quantized_node

                Args:
                    model: floating point torch.nn.Module
                    inputs: one positional sample input arguments for model
                    expected_node: NodeSpec
                        e.g. NodeSpec.call_function(torch.quantize_per_tensor)
                    expected_node_occurrence: a dict from NodeSpec to
                        expected number of occurrences (int)
                        e.g. {NodeSpec.call_function(torch.quantize_per_tensor) : 1,
                                NodeSpec.call_method('dequantize'): 1}
                    expected_node_list: a list of NodeSpec, used to check the order
                        of the occurrence of Node
                        e.g. [NodeSpec.call_function(torch.quantize_per_tensor),
                                NodeSpec.call_module(nnq.Conv2d),
                                NodeSpec.call_function(F.hardtanh_),
                                NodeSpec.call_method('dequantize')]
                    is_reference: if True, enables reference mode
                    print_debug_info: if True, prints debug info
                    custom_qconfig_dict: overrides default qconfig_dict
                    prepare_expected_node: same as expected_node, but for prepare
                    prepare_expected_node_occurrence: same as
                        expected_node_occurrence, but for prepare
                    prepare_expected_node_list: same as expected_node_list, but
                        for prepare

                Returns:
                    A dictionary with the following structure:
                   {
                       "prepared": ...,  # the prepared model
                       "quantized": ...,  # the quantized non-reference model
                       "quantized_reference": ...,  # the quantized reference model
                       "result": ...,  # the result for either quantized or
                                       # quantized_reference model depending on the
                                       # is_reference argument
                   }
            """
    if type(inputs) == list:
        inputs = inputs[0]
    if quant_type == QuantType.QAT:
        qconfig_mapping = get_default_qat_qconfig_mapping(torch.backends.quantized.engine)
        model.train()
    elif quant_type == QuantType.STATIC:
        qconfig_mapping = get_default_qconfig_mapping(torch.backends.quantized.engine)
        model.eval()
    else:
        qconfig = default_dynamic_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        model.eval()
    if quant_type == QuantType.QAT:
        prepare = prepare_qat_fx
    else:
        prepare = prepare_fx
    if custom_qconfig_dict is not None:
        assert type(custom_qconfig_dict) in (QConfigMapping, dict), 'custom_qconfig_dict should be a QConfigMapping or a dict'
        if isinstance(custom_qconfig_dict, QConfigMapping):
            qconfig_mapping = custom_qconfig_dict
        else:
            qconfig_mapping = QConfigMapping.from_dict(custom_qconfig_dict)
    prepared = prepare(model, qconfig_mapping, example_inputs=inputs, prepare_custom_config=prepare_custom_config, backend_config=backend_config)
    if not quant_type == QuantType.DYNAMIC:
        prepared(*inputs)
    if print_debug_info:
        print()
        print('quant type:\n', quant_type)
        print('original model:\n', model)
        print()
        print('prepared model:\n', prepared)
    self.checkGraphModuleNodes(prepared, prepare_expected_node, prepare_expected_node_occurrence, prepare_expected_node_list)
    prepared_copy = copy.deepcopy(prepared)
    qgraph = convert_fx(copy.deepcopy(prepared))
    qgraph_reference = convert_to_reference_fx(copy.deepcopy(prepared))
    result = qgraph(*inputs)
    result_reference = qgraph_reference(*inputs)
    qgraph_copy = copy.deepcopy(qgraph)
    qgraph_reference_copy = copy.deepcopy(qgraph_reference)
    qgraph_to_check = qgraph_reference if is_reference else qgraph
    if print_debug_info:
        print()
        print('quantized model:\n', qgraph_to_check)
        self.printGraphModule(qgraph_to_check)
        print()
    self.checkGraphModuleNodes(qgraph_to_check, expected_node, expected_node_occurrence, expected_node_list)
    return {'prepared': prepared_copy, 'quantized': qgraph_copy, 'quantized_reference': qgraph_reference_copy, 'quantized_output': result, 'quantized_reference_output': result_reference}