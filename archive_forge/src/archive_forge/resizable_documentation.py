from typing import Callable, Optional, TypeVar
from ..config import registry
from ..model import Model
from ..types import Floats2d
Create a resized copy of a layer that has parameters W and b and dimensions nO and nI.