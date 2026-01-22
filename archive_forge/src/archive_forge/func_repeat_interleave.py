import functools
import torch
import torch._C._onnx as _C_onnx
from torch.onnx import (
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::repeat_interleave')
@_beartype.beartype
def repeat_interleave(g: jit_utils.GraphContext, self, repeats, dim=None, output_size=None):
    input = self
    final_dim = dim
    if symbolic_helper._is_none(dim):
        input = symbolic_helper._reshape_helper(g, self, g.op('Constant', value_t=torch.tensor([-1])))
        dim = torch.tensor(0, dtype=torch.int64)
    else:
        dim = symbolic_helper._maybe_get_scalar(dim)
    repeats_dim = symbolic_helper._get_tensor_rank(repeats)
    repeats_sizes = symbolic_helper._get_tensor_sizes(repeats)
    input_sizes = symbolic_helper._get_tensor_sizes(input)
    if repeats_dim is None:
        raise errors.SymbolicValueError('Unsupported: ONNX export of repeat_interleave for unknown repeats rank.', self)
    if repeats_sizes is None:
        raise errors.SymbolicValueError('Unsupported: ONNX export of repeat_interleave for unknown repeats size.', self)
    if input_sizes is None:
        raise errors.SymbolicValueError('Unsupported: ONNX export of repeat_interleave for unknown input size.', self)
    if dim < 0:
        dim += len(input_sizes)
    output_sizes = input_sizes.copy()
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            output_sizes[idx], input_sizes[idx] = (0, -1)
    if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
        return symbolic_helper._repeat_interleave_single_value_repeat_helper(g, self, repeats, dim)
    cond_dynamic_repeats = repeats_dim == 1 and repeats_sizes[0] is None
    if output_sizes[dim] == 0 or cond_dynamic_repeats:
        reps = symbolic_helper._size_helper(g, input, dim)
        reps = opset11.unsqueeze(g, reps, 0)
        if cond_dynamic_repeats:
            repeat_dim = symbolic_helper._size_helper(g, repeats, g.op('Constant', value_t=torch.LongTensor([0])))
            repeat_cond = g.op('Equal', repeat_dim, g.op('Constant', value_t=torch.LongTensor([1])))
            repeats = where(g, repeat_cond, g.op('Expand', repeats, reps), repeats)
    else:
        return opset9.repeat_interleave(g, self, repeats, final_dim)
    reps_like = g.op('ConstantOfShape', g.op('Shape', repeats), value_t=torch.tensor([1], dtype=torch.long))
    r_splits = split(g, repeats, reps_like, 0)
    i_splits = split(g, input, reps_like, dim)
    output_sizes[dim], input_sizes[dim] = (-1, 1)
    loop_condition = g.op('Constant', value_t=torch.tensor(1))
    loop_condition = g.op('Cast', loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    loop_len = reps
    final_splits = g.op('SequenceEmpty')
    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(g, 'Loop', loop_len, loop_condition, final_splits, n_blocks=1)
    loop_block = loop_context.block
    block_input_iter = utils._add_input_to_block(loop_block)
    cond = utils._add_input_to_block(loop_block)
    final_splits = utils._add_input_to_block(loop_block)
    r_split = loop_context.op('SequenceAt', r_splits, block_input_iter)
    i_split = loop_context.op('SequenceAt', i_splits, block_input_iter)
    i_split = opset11.unsqueeze(loop_context, i_split, dim + 1)
    r_concat = [loop_context.op('Constant', value_t=torch.LongTensor(input_sizes[:dim + 1])), r_split, loop_context.op('Constant', value_t=torch.LongTensor(input_sizes[dim + 1:]))]
    r_concat = loop_context.op('Concat', *r_concat, axis_i=0)
    i_split = opset9.expand(loop_context, i_split, r_concat, None)
    i_split = symbolic_helper._reshape_helper(loop_context, i_split, g.op('Constant', value_t=torch.LongTensor(output_sizes)))
    final_splits = loop_context.op('SequenceInsert', final_splits, i_split)
    cond_out = loop_context.op('Cast', loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    utils._add_output_to_block(loop_block, cond_out)
    utils._add_output_to_block(loop_block, final_splits)
    loop_out = loop.node().output()
    loop_out = g.op('ConcatFromSequence', loop_out, axis_i=dim)
    return loop_out