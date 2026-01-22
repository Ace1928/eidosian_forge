from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_onnx_symbolic('prim::If')
@_beartype.beartype
def prim_if(g: jit_utils.GraphContext, *inputs, **attrs) -> List[_C.Value]:
    n = g.original_node
    block = g.block
    env = g.env
    params_dict = g.params_dict
    operator_export_type = GLOBALS.operator_export_type
    opset_version = GLOBALS.export_onnx_opset_version
    static_if = inputs[0].node().kind() == 'onnx::Constant'
    if static_if:
        input_flag = symbolic_helper._node_get(inputs[0].node(), 'value').tolist()
        const_value = all(input_flag) if isinstance(input_flag, list) else bool(input_flag)
        block_idx = 0 if const_value else 1
        current_b = list(n.blocks())[block_idx]
        env = torch._C._jit_pass_onnx_block(current_b, block, operator_export_type, env, True)
        if_output_list = list(n.outputs())
        current_b_list = list(current_b.outputs())
        final_b_list = []
        for idx in range(len(if_output_list)):
            if current_b_list[idx] not in env:
                raise errors.SymbolicValueError(f'The sub block ATen output {current_b_list[idx]} is not in env.', current_b_list[idx])
            onnx_b = env[current_b_list[idx]]
            final_b_list.append(onnx_b)
        return final_b_list
    else:
        old_blocks = tuple(n.blocks())
        new_op_outputs, new_block_contexts, new_node = jit_utils.add_op_with_blocks(g, 'If', *inputs, outputs=n.outputsSize(), n_blocks=len(old_blocks))
        for old_block, new_block_context in zip(old_blocks, new_block_contexts):
            torch._C._jit_pass_onnx_block(old_block, new_block_context.block, operator_export_type, env, False)
        fixed_outputs = torch._C._jit_pass_fixup_onnx_controlflow_node(new_node, opset_version)
        if GLOBALS.onnx_shape_inference:
            torch._C._jit_pass_onnx_node_shape_type_inference(new_node, params_dict, opset_version)
        return fixed_outputs