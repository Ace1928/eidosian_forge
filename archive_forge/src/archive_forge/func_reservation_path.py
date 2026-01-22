import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
def reservation_path(self) -> str:
    return self._reservation_format_str % dict(self)