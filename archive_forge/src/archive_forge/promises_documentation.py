import inspect
import sys
from collections import deque
from weakref import WeakMethod, ref
from .abstract import Thenable
from .utils import reraise
Return the callable or a weak reference.

        Handles both bound and unbound methods.
        