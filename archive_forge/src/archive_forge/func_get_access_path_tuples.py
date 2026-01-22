import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
def get_access_path_tuples(self):
    accesses = [create_access(self._inference_state, o) for o in self._get_objects_path()]
    return [(access.py__name__(), access) for access in accesses]