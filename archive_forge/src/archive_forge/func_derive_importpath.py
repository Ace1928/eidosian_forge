from contextlib import contextmanager
import os
import re
import sys
from typing import Any
from typing import final
from typing import Generator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
from typing import Union
import warnings
from _pytest.fixtures import fixture
from _pytest.warning_types import PytestWarning
def derive_importpath(import_path: str, raising: bool) -> Tuple[str, object]:
    if not isinstance(import_path, str) or '.' not in import_path:
        raise TypeError(f'must be absolute import path string, not {import_path!r}')
    module, attr = import_path.rsplit('.', 1)
    target = resolve(module)
    if raising:
        annotated_getattr(target, attr, ann=module)
    return (attr, target)