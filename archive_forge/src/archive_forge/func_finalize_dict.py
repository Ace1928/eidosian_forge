from .zstdfile import *
from .seekable_zstdfile import *
def finalize_dict(zstd_dict, samples, dict_size, level):
    """Finalize a zstd dictionary, return a ZstdDict object.

    Given a custom content as a basis for dictionary, and a set of samples,
    finalize dictionary by adding headers and statistics according to the zstd
    dictionary format.

    You may compose an effective dictionary content by hand, which is used as
    basis dictionary, and use some samples to finalize a dictionary. The basis
    dictionary can be a "raw content" dictionary, see is_raw parameter in
    ZstdDict.__init__ method.

    Parameters
    zstd_dict: A ZstdDict object, basis dictionary.
    samples:   An iterable of samples, a sample is a bytes-like object
               represents a file.
    dict_size: The dictionary's maximum size, in bytes.
    level:     The compression level expected to use in production. The
               statistics for each compression level differ, so tuning the
               dictionary for the compression level can help quite a bit.
    """
    if zstd_version_info < (1, 4, 5):
        msg = "This function only available when the underlying zstd library's version is greater than or equal to v1.4.5, the current underlying zstd library's version is v%s." % zstd_version
        raise NotImplementedError(msg)
    if not isinstance(zstd_dict, ZstdDict):
        raise TypeError('zstd_dict argument should be a ZstdDict object.')
    if not isinstance(dict_size, int):
        raise TypeError('dict_size argument should be an int object.')
    if not isinstance(level, int):
        raise TypeError('level argument should be an int object.')
    chunks = []
    chunk_sizes = []
    for chunk in samples:
        chunks.append(chunk)
        chunk_sizes.append(_nbytes(chunk))
    chunks = b''.join(chunks)
    if not chunks:
        raise ValueError("The samples are empty content, can't finalize dictionary.")
    dict_content = _finalize_dict(zstd_dict.dict_content, chunks, chunk_sizes, dict_size, level)
    return ZstdDict(dict_content)