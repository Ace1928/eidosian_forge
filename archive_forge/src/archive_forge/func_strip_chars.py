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
@udf(STRING)
def strip_chars(haystack, chars):
    return haystack.strip(chars)