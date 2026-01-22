import logging
import os.path
def register_compressor(ext, callback):
    """Register a callback for transparently decompressing files with a specific extension.

    Parameters
    ----------
    ext: str
        The extension.  Must include the leading period, e.g. ``.gz``.
    callback: callable
        The callback.  It must accept two position arguments, file_obj and mode.
        This function will be called when ``smart_open`` is opening a file with
        the specified extension.

    Examples
    --------

    Instruct smart_open to use the `lzma` module whenever opening a file
    with a .xz extension (see README.rst for the complete example showing I/O):

    >>> def _handle_xz(file_obj, mode):
    ...     import lzma
    ...     return lzma.LZMAFile(filename=file_obj, mode=mode, format=lzma.FORMAT_XZ)
    >>>
    >>> register_compressor('.xz', _handle_xz)

    """
    if not (ext and ext[0] == '.'):
        raise ValueError('ext must be a string starting with ., not %r' % ext)
    ext = ext.lower()
    if ext in _COMPRESSOR_REGISTRY:
        logger.warning('overriding existing compression handler for %r', ext)
    _COMPRESSOR_REGISTRY[ext] = callback