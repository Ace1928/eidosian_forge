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
def register_aggregate_groups(db, *groups):
    seen = set()
    for group in groups:
        klasses = AGGREGATE_COLLECTION.get(group, ())
        for klass in klasses:
            name = getattr(klass, 'name', klass.__name__)
            if name not in seen:
                seen.add(name)
                db.register_aggregate(klass, name)