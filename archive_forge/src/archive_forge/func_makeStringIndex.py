import string
from datetime import datetime
import numpy as np
import pandas as pd
def makeStringIndex(k=10, name=None):
    return pd.Index(rands_array(nchars=10, size=k), name=name)