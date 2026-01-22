from typing import Callable, Iterable, Iterator, List, Optional

        >>> f = IterableFile([b'Th\nis ', b'is \n', b'a ', b'te\nst.'])
        >>> f.readlines()
        [b'Th\n', b'is is \n', b'a te\n', b'st.']
        >>> f = IterableFile([b'Th\nis ', b'is \n', b'a ', b'te\nst.'])
        >>> f.close()
        >>> f.readlines()
        Traceback (most recent call last):
        ValueError: File is closed.
        