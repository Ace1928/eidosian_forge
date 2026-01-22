from subprocess import Popen, PIPE
from distutils import spawn
import os
import math
import random
import time
import sys
import platform
def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number