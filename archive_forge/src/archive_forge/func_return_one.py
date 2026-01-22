import platform
import time
import unittest
import pytest
from monty.functools import (
@return_if_raise(ValueError, 'hello')
def return_one(self):
    return 1