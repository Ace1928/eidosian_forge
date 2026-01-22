import json
import pathlib
import warnings
from typing import IO, Union, Optional, Literal
from .mimebundle import spec_to_mimebundle
from ..vegalite.v5.data import data_transformers
from altair.utils._vegafusion_data import using_vegafusion
def write_file_or_filename(fp: Union[str, pathlib.PurePath, IO], content: Union[str, bytes], mode: str='w', encoding: Optional[str]=None) -> None:
    """Write content to fp, whether fp is a string, a pathlib Path or a
    file-like object"""
    if isinstance(fp, str) or isinstance(fp, pathlib.PurePath):
        with open(file=fp, mode=mode, encoding=encoding) as f:
            f.write(content)
    else:
        fp.write(content)