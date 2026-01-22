from typing import Callable, List, Optional, Tuple, Union
import math
import warnings
import importlib
import torch
from torch import _VF
from torch import sym_int as _sym_int
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes, sparse_support_notes
from typing import TYPE_CHECKING
from .._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
from ..overrides import (
from . import _reduction as _Reduction
from . import grad  # noqa: F401
from .modules import utils
from .modules.utils import _single, _pair, _triple, _list_with_default
def embedding_bag(input: Tensor, weight: Tensor, offsets: Optional[Tensor]=None, max_norm: Optional[float]=None, norm_type: float=2, scale_grad_by_freq: bool=False, mode: str='mean', sparse: bool=False, per_sample_weights: Optional[Tensor]=None, include_last_offset: bool=False, padding_idx: Optional[int]=None) -> Tensor:
    """Compute sums, means or maxes of `bags` of embeddings.

    Calculation is done without instantiating the intermediate embeddings.
    See :class:`torch.nn.EmbeddingBag` for more details.

    Note:
        {backward_reproducibility_note}

    Args:
        input (LongTensor): Tensor containing bags of indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        offsets (LongTensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                             the starting index position of each bag (sequence) in :attr:`input`.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The ``p`` in the ``p``-norm to compute for the :attr:`max_norm` option.
                                     Default ``2``.
        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (str, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.
                                 Note: this option is not supported when ``mode="max"``.
        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be 1. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not None.

        include_last_offset (bool, optional): if ``True``, the size of offsets is equal to the number of bags + 1.
            The last element is the size of the input, or the ending index position of the last bag (sequence).

        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the
                                     gradient; therefore, the embedding vector at :attr:`padding_idx` is not updated
                                     during training, i.e. it remains as a fixed "pad". Note that the embedding
                                     vector at :attr:`padding_idx` is excluded from the reduction.

    Shape:
        - :attr:`input` (LongTensor) and :attr:`offsets` (LongTensor, optional)

          - If :attr:`input` is 2D of shape `(B, N)`, it will be treated as ``B`` bags (sequences)
            each of fixed length ``N``, and this will return ``B`` values aggregated in a way
            depending on the :attr:`mode`. :attr:`offsets` is ignored and required to be ``None`` in this case.

          - If :attr:`input` is 1D of shape `(N)`, it will be treated as a concatenation of
            multiple bags (sequences). :attr:`offsets` is required to be a 1D tensor containing
            the starting index positions of each bag in :attr:`input`. Therefore, for :attr:`offsets`
            of shape `(B)`, :attr:`input` will be viewed as having ``B`` bags.
            Empty bags (i.e., having 0-length) will have returned vectors filled by zeros.

        - :attr:`weight` (Tensor): the learnable weights of the module of shape `(num_embeddings, embedding_dim)`

        - :attr:`per_sample_weights` (Tensor, optional). Has the same shape as :attr:`input`.

        - :attr:`output`: aggregated embedding values of shape `(B, embedding_dim)`

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        >>> offsets = torch.tensor([0, 4])
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> F.embedding_bag(input, embedding_matrix, offsets)
        tensor([[ 0.3397,  0.3552,  0.5545],
                [ 0.5893,  0.4386,  0.5882]])

        >>> # example with padding_idx
        >>> embedding_matrix = torch.rand(10, 3)
        >>> input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9])
        >>> offsets = torch.tensor([0, 4])
        >>> F.embedding_bag(input, embedding_matrix, offsets, padding_idx=2, mode='sum')
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7082,  3.2145, -2.6251]])
    """
    if has_torch_function_variadic(input, weight, offsets, per_sample_weights):
        return handle_torch_function(embedding_bag, (input, weight, offsets, per_sample_weights), input, weight, offsets=offsets, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, mode=mode, sparse=sparse, per_sample_weights=per_sample_weights, include_last_offset=include_last_offset, padding_idx=padding_idx)
    if weight.dtype == torch.long and input.is_floating_point():
        warnings.warn('Argument order of nn.functional.embedding_bag was changed. Usage `embedding_bag(weight, input, ...)` is deprecated, and should now be `embedding_bag(input, weight, ...)`.')
        weight, input = (input, weight)
    if per_sample_weights is not None and input.size() != per_sample_weights.size():
        raise ValueError(f'embedding_bag: If per_sample_weights ({per_sample_weights.shape}) is not None, then it must have the same shape as the input ({input.shape})')
    if not weight.dim() == 2:
        raise ValueError(f'weight has to be a 2D Tensor, but got Tensor of dimension {weight.dim()}')
    if input.dim() == 2:
        if offsets is not None:
            type_str = '<unknown>'
            if not torch.jit.is_scripting():
                type_str = str(type(offsets))
            raise ValueError(f'if input is 2D, then offsets has to be None, as input is treated is a mini-batch of fixed length sequences. However, found offsets of type {type_str}')
        offsets = torch.arange(0, input.numel(), input.size(1), dtype=input.dtype, device=input.device)
        input = input.reshape(-1)
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.reshape(-1)
    elif input.dim() == 1:
        if offsets is None:
            raise ValueError('offsets has to be a 1D Tensor but got None')
        if offsets.dim() != 1:
            raise ValueError('offsets has to be a 1D Tensor')
    else:
        raise ValueError(f'input has to be 1D or 2D Tensor, but got Tensor of dimension {input.dim()}')
    if mode == 'sum':
        mode_enum = 0
    elif mode == 'mean':
        mode_enum = 1
    elif mode == 'max':
        mode_enum = 2
        if scale_grad_by_freq:
            raise ValueError('max mode does not support scaling the gradient by the frequency')
        if sparse:
            raise ValueError('max mode does not support sparse weights')
    else:
        raise ValueError('mode has to be one of sum, mean or max')
    if max_norm is not None:
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    if per_sample_weights is not None and mode != 'sum':
        raise NotImplementedError(f"embedding_bag: per_sample_weights was not None. per_sample_weights is only supported for mode='sum' (got mode='{mode}'). Please open a feature request on GitHub.")
    ret, _, _, _ = torch.embedding_bag(weight, input, offsets, scale_grad_by_freq, mode_enum, sparse, per_sample_weights, include_last_offset, padding_idx)
    return ret