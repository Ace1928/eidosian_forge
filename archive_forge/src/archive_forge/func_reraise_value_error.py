import platform
import time
import unittest
import pytest
from monty.functools import (
@return_if_raise(KeyError, 'hello')
def reraise_value_error(self):
    raise ValueError()