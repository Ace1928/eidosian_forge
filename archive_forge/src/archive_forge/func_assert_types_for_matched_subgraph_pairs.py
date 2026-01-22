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
def assert_types_for_matched_subgraph_pairs(self, matched_subgraph_pairs: Dict[str, Tuple[NSSubgraph, NSSubgraph]], expected_types: Dict[str, Tuple[Tuple[Callable, Callable], Tuple[Callable, Callable]]], gm_a: GraphModule, gm_b: GraphModule) -> None:
    """
            Verifies that the types specified in expected_types match
            the underlying objects pointed to by the nodes in matched_subgraph_pairs.

            An example successful test case:

              matched_subgraph_pairs = {'x0': (graph_a_conv_0_node, graph_b_conv_0_node)}
              expected_types = {'x0': (nn.Conv2d, nnq.Conv2d)}

            The function tests for key equivalence, and verifies types with
            instance checks.
            """

    def _get_underlying_op_type(node: Node, gm: GraphModule) -> Union[Callable, str]:
        if node.op == 'call_module':
            mod = getattr(gm, node.target)
            return type(mod)
        else:
            assert node.op in ('call_function', 'call_method')
            return node.target
    self.assertTrue(len(matched_subgraph_pairs) == len(expected_types), f'Expected length of results to match, but got {len(matched_subgraph_pairs)} and {len(expected_types)}')
    for k, v in expected_types.items():
        expected_types_a, expected_types_b = v
        exp_type_start_a, exp_type_end_a = expected_types_a
        exp_type_start_b, exp_type_end_b = expected_types_b
        subgraph_a, subgraph_b = matched_subgraph_pairs[k]
        act_type_start_a = _get_underlying_op_type(subgraph_a.start_node, gm_a)
        act_type_start_b = _get_underlying_op_type(subgraph_b.start_node, gm_b)
        act_type_end_a = _get_underlying_op_type(subgraph_a.end_node, gm_a)
        act_type_end_b = _get_underlying_op_type(subgraph_b.end_node, gm_b)
        types_match = exp_type_start_a is act_type_start_a and exp_type_end_a is act_type_end_a and (exp_type_start_b is act_type_start_b) and (exp_type_end_b is act_type_end_b)
        self.assertTrue(types_match, 'Type mismatch at {}: expected {}, got {}'.format(k, (exp_type_start_a, exp_type_end_a, exp_type_start_b, exp_type_end_b), (act_type_start_a, act_type_end_a, act_type_start_b, act_type_end_b)))