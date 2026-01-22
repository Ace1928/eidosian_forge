import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
@classmethod
def class_method(cls, a, b=10, *, c):
    pass