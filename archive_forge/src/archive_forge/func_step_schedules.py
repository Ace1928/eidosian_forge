import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, cast
from .backends import get_array_ops
from .config import registry
from .types import FloatsXd, Generator
def step_schedules(self):
    for key, schedule in self.schedules.items():
        try:
            value = next(schedule)
        except StopIteration:
            value = getattr(self, key)
        setattr(self, key, value)