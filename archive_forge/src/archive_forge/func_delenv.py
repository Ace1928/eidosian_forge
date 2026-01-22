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
def delenv(self, name: str, raising: bool=True) -> None:
    """Delete ``name`` from the environment.

        Raises ``KeyError`` if it does not exist, unless ``raising`` is set to
        False.
        """
    environ: MutableMapping[str, str] = os.environ
    self.delitem(environ, name, raising=raising)