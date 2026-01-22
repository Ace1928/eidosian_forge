from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
backslashes aren't allowed in datashapes according to the definitions
        in lexer.py as of 2014-10-02. This is probably an oversight that should
        be fixed.
        