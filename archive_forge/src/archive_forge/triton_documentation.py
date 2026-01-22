from dataclasses import replace
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple
import torch
from ... import _is_triton_available
from ..common import register_operator
from .attn_bias import LowerTriangularMask
from .common import (
Import a module from the given path, w/o __init__.py