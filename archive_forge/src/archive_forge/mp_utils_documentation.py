import math
import fileio
import collections
import platform
import multiprocessing as multiproc
import random
from functools import reduce
from itertools import chain, count, islice, takewhile
from typing import List, Optional, Dict

    Compose all the function arguments together
    :param functions: Functions to compose
    :return: Single composed function
    