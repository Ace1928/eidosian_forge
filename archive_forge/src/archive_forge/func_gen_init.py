import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
def gen_init():
    yield (m.x[1], 10)
    yield (m.x[2], 20)
    yield (m.x[3], 30)
    yield (m.y, 100)