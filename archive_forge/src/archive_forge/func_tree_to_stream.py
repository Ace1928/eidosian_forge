from stat import S_ISDIR
from git.compat import safe_decode, defenc
from typing import (
def tree_to_stream(entries: Sequence[EntryTup], write: Callable[['ReadableBuffer'], Union[int, None]]) -> None:
    """Write the given list of entries into a stream using its write method.

    :param entries: **sorted** list of tuples with (binsha, mode, name)
    :param write: write method which takes a data string
    """
    ord_zero = ord('0')
    bit_mask = 7
    for binsha, mode, name in entries:
        mode_str = b''
        for i in range(6):
            mode_str = bytes([(mode >> i * 3 & bit_mask) + ord_zero]) + mode_str
        if mode_str[0] == ord_zero:
            mode_str = mode_str[1:]
        if isinstance(name, str):
            name_bytes = name.encode(defenc)
        else:
            name_bytes = name
        write(b''.join((mode_str, b' ', name_bytes, b'\x00', binsha)))