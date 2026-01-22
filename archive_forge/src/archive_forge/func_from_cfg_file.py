import configparser
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import Union
from .helpers import make_path
@classmethod
def from_cfg_file(cls, filespec: Union[Path, str]):
    cfg = _parse_cfg_file(filespec)
    dispatch = _build_getter_dispatch(cfg, cls.section_header, converters=cls.converters)
    kwargs = {field.name: dispatch.get_typed_value(field.type, field.name) for field in dataclasses.fields(cls)}
    return cls(**kwargs)