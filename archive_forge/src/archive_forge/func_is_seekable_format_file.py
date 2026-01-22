from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
@staticmethod
def is_seekable_format_file(filename):
    """Check if a file is Zstandard Seekable Format file or 0-size file.

        It parses the seek table at the end of the file, returns True if no
        format error.

        filename can be either a file path (str/bytes/PathLike), or can be an
        existing file object in reading mode.
        """
    if isinstance(filename, (str, bytes, PathLike)):
        fp = io.open(filename, 'rb')
        is_file_path = True
    elif hasattr(filename, 'readable') and filename.readable() and hasattr(filename, 'seekable') and filename.seekable():
        fp = filename
        is_file_path = False
        orig_pos = fp.tell()
    else:
        raise TypeError('filename argument should be a str/bytes/PathLike object, or a file object that is readable and seekable.')
    table = SeekTable(read_mode=False)
    try:
        table.load_seek_table(fp, seek_to_0=False)
    except SeekableFormatError:
        ret = False
    else:
        ret = True
    finally:
        if is_file_path:
            fp.close()
        else:
            fp.seek(orig_pos)
    return ret