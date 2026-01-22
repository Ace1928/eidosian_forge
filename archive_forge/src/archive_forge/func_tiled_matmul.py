import os
from typing import List, Optional
import torch
import torch.multiprocessing.reductions
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing_extensions import Annotated
from .. import _is_triton_available
from .common import Alias, make_pytorch_operator_for_dispatch_key
def tiled_matmul(a: List[List[torch.Tensor]], b: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Multiply two matrices given as grids of tiles

    It performs the matmul between A and B, which are given as two-dimensional
    grids of tiles (i.e., blocks), represented as lists of lists of tensors.
    The output will itself be a matrix in such a form. Formally:

        out[m][n] = sum(a[m][k] @ b[k][n] for k in range(...))

    with the obvious constraints needed to make it work, in terms of number of
    tiles and sizes of each tile.

    The interest of this operator is to improve performance by avoding wave
    quantization effects when doing independent matrix multiplications in
    series. Sometimes, when these matmuls have one operand in common, this can
    also be addressed by concatenating the other operands into a single matrix,
    and issuing a single matmul. However this isn't always possible (e.g., might
    break the checkpoint format) and it's an anti-pattern, as it obscures the
    logic (e.g., changing the modelling code out of performance reasons). This
    tiled matmul performs the same computation as if the matrices were merged,
    without merging them, simply through a smarter memory addressing scheme.

    The tiled matmul is less generic than a grouped matmul, which can also help
    with wave quantization, and doesn't need the matmuls to have the same lhs
    or rhs operand. However, a grouped matmul will write the result of each
    matmul to a separate output matrix, whereas the tiled matmul allows to add
    them together into a single output. This is needed during the backward pass
    of a linear layer, and it's the reason we wrote this instead of using a
    grouped matmul.

    The tiled matmul is implemented using a custom Triton kernel, which puts
    constraints on the strides of the tiles. All rows of A must have the same
    K stride, all columns of A must have the same M stride, and so on.

    Currently the tiled matmul supports at most three tiles on each dimension,
    although fewer can also be given. This is because we needed it to fuse the
    query, key and value weights of an attention layer. This limit can be
    increased if needed.

    This operator is differentiable.

    """
    ab_tree_values, ab_tree_spec = tree_flatten((a, b))
    c_tree_spec, *c_tree_values = _TiledMatmul.apply(ab_tree_spec, *ab_tree_values)
    c = tree_unflatten(list(c_tree_values), c_tree_spec)
    return c