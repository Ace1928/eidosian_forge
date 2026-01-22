import platform
import time
import unittest
import pytest
from monty.functools import (
@return_if_raise([KeyError, ValueError], 'hello')
def catch_exc_list(self):
    import random
    if random.randint(0, 1) == 0:
        raise ValueError()
    else:
        raise KeyError()