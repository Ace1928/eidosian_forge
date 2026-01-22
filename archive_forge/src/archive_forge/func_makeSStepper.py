import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import numpy as np
from .AxisItem import AxisItem
def makeSStepper(stepSize):

    def stepper(val, n, first: bool):
        if val < MIN_REGULAR_TIMESTAMP or val > MAX_REGULAR_TIMESTAMP:
            return np.inf
        if first:
            return (val // (n * stepSize) + 1) * (n * stepSize)
        else:
            return val + n * stepSize
    return stepper