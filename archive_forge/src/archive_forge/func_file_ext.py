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
@udf(FILE)
def file_ext(filename):
    try:
        res = os.path.splitext(filename)
    except ValueError:
        return None
    return res[1]