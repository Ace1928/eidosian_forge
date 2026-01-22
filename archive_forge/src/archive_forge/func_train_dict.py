from .zstdfile import *
from .seekable_zstdfile import *
def train_dict(samples, dict_size):
    """Train a zstd dictionary, return a ZstdDict object.

    Parameters
    samples:   An iterable of samples, a sample is a bytes-like object
               represents a file.
    dict_size: The dictionary's maximum size, in bytes.
    """
    if not isinstance(dict_size, int):
        raise TypeError('dict_size argument should be an int object.')
    chunks = []
    chunk_sizes = []
    for chunk in samples:
        chunks.append(chunk)
        chunk_sizes.append(_nbytes(chunk))
    chunks = b''.join(chunks)
    if not chunks:
        raise ValueError("The samples are empty content, can't train dictionary.")
    dict_content = _train_dict(chunks, chunk_sizes, dict_size)
    return ZstdDict(dict_content)