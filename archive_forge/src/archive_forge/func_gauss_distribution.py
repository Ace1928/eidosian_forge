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
def gauss_distribution(mean, sigma):
    try:
        return random.gauss(mean, sigma)
    except ValueError:
        return None