import configparser
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import Union
from .helpers import make_path
def get_typed_value(self, type_, name):
    get_fn = self.get_fn_for_type(type_)
    return get_fn(name)