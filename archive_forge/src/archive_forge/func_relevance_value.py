import sys
import os
import re
from numpy import ufunc
def relevance_value(a):
    return relevance(a, *cache[a])