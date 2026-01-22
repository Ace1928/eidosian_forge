import json
import re
import warnings
import numpy as np
import pandas as pd
import pandas.io.formats.format as fmt
def n_suffix(matchobj):
    return 'BigInt("' + matchobj.group(1) + '")' + matchobj.group(2)