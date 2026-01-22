import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
@is_maximize.setter
def is_maximize(self, is_maximize: bool) -> None:
    self.model.storage.set_is_maximize(is_maximize)