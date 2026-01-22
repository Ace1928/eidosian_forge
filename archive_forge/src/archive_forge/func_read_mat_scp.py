from typing import Any, Callable, Iterable, Tuple
import torch
from torch import Tensor
from torchaudio._internal import module_utils as _mod_utils
@_mod_utils.requires_module('kaldi_io', 'numpy')
def read_mat_scp(file_or_fd: Any) -> Iterable[Tuple[str, Tensor]]:
    """Create generator of (key,matrix<float32/float64>) tuples, read according to Kaldi scp.

    Args:
        file_or_fd (str/FileDescriptor): scp, gzipped scp, pipe or opened file descriptor

    Returns:
        Iterable[Tuple[str, Tensor]]: The string is the key and the tensor is the matrix read from file

    Example
        >>> # read scp to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_mat_scp(file) }
    """
    import kaldi_io
    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_mat_scp)