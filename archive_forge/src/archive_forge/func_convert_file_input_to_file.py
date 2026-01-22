from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
@classmethod
def convert_file_input_to_file(cls, data: InputContentType, path: Optional[InputPathType]=None, make_temp: Optional[bool]=False, **kwargs) -> 'FileLike':
    """
        Converts input to a file
        """
    if isinstance(data, bytes):
        return cls.convert_bytes_to_file(data, path=path, make_temp=make_temp)
    if isinstance(data, Path) or hasattr(data, 'as_posix'):
        return cls.convert_bytes_to_file(data.read_bytes()) if make_temp else data
    if isinstance(data, str):
        p = File(data)
        if p.exists():
            return cls.convert_bytes_to_file(p.read_bytes()) if make_temp else p
        return cls.convert_text_to_file(data, path=path, make_temp=make_temp)
    if isinstance(data, Iterable):
        return cls.convert_bytes_to_file(b''.join(data), path=path, make_temp=make_temp)
    raise TypeError(f'Invalid data type: {type(data)}')