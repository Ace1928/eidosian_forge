from typing import Callable, Iterable, Iterator, List, Optional
def read_n(self, length: int) -> bytes:
    """
        >>> IterableFileBase([b'This ', b'is ', b'a ', b'test.']).read_n(8)
        b'This is '
        """

    def test_length(result):
        if len(result) >= length:
            return length
        else:
            return None
    return self._read(test_length)