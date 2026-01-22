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
@udf(DATE)
def human_delta(nseconds, glue=', '):
    parts = ((86400 * 365, 'year'), (86400 * 30, 'month'), (86400 * 7, 'week'), (86400, 'day'), (3600, 'hour'), (60, 'minute'), (1, 'second'))
    accum = []
    for offset, name in parts:
        val, nseconds = divmod(nseconds, offset)
        if val:
            suffix = val != 1 and 's' or ''
            accum.append('%s %s%s' % (val, name, suffix))
    if not accum:
        return '0 seconds'
    return glue.join(accum)