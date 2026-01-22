import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def is_intent(self, *names):
    for name in names:
        if name not in self.intent_list:
            return False
    return True