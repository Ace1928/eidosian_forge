import time
from math import sqrt
from os.path import isfile
from ase.io.jsonio import read_json, write_json
from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import world, barrier
from ase.io.trajectory import Trajectory
from ase.utils import IOContext
import collections.abc
def insert_observer(self, function, position=0, interval=1, *args, **kwargs):
    """Insert an observer."""
    if not isinstance(function, collections.abc.Callable):
        function = function.write
    self.observers.insert(position, (function, interval, args, kwargs))