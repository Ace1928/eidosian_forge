import binascii
import lzma
import platform
import sys
def get_default_blocksize() -> int:
    """
    Return a safe buffer size for pass to decompress/compress modules.
    We check several conditions;

    1. 32bit python
      It sometimes fails with memory_error
      when decompress/compress large file (hundreds of MiB) on 32bit python.
      We reduce buffer size smaller to avoid memory_error.
      @see issue https://github.com/miurahr/py7zr/issues/370

    2. CPython
      CPython implementation of 3.7.5 fixed a lzma module bug
      that is not respect max_length parameter.
      When buffer size is larger than default of the module,
      max_length can be a value to lead the bug, so we set it
      lzma module default = 32768 bytes.
      @see BPO-21872: LZMA library sometimes fails to decompress a file
           https://bugs.python.org/issue21872
      @see https://github.com/miurahr/py7zr/issues/272

    3. PyPy
      PyPy 7.2 (python 3.6.9) fixed a lzma module's bug as same as
      CPython above.
      We set buffer size is as default size of the module to avoid the bugs.
      @see PyPy3-3090: lzma.LZMADecomporessor.decompress does not respect max_length
           https://foss.heptapod.net/pypy/pypy/-/issues/3090
      @see https://github.com/miurahr/py7zr/pull/114

    :return: recommended buffer size as int.
    """
    if is_64bit() and (is_pypy369later() or sys.version_info >= (3, 7, 5)):
        return 1048576
    else:
        return 32768