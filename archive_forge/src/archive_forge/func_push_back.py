from typing import Callable, Iterable, Iterator, List, Optional
def push_back(self, contents: bytes) -> None:
    """
        >>> f = IterableFileBase([b'Th\\nis ', b'is \\n', b'a ', b'te\\nst.'])
        >>> f.read_to(b'\\n')
        b'Th\\n'
        >>> f.push_back(b"Sh")
        >>> f.read_all()
        b'Shis is \\na te\\nst.'
        """
    self._buffer = contents + self._buffer