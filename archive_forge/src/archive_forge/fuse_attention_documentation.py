import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (

    Equivalent to functools.partial but also updates the signature on returned function
    