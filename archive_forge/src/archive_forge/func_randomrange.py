import datetime
import hashlib
import heapq
import math
import os
import random
import re
import sys
import threading
import zlib
from peewee import format_date_time
@udf(MATH)
def randomrange(start, end=None, step=None):
    if end is None:
        start, end = (0, start)
    elif step is None:
        step = 1
    return random.randrange(start, end, step)