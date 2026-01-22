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
def register_table_function_groups(db, *groups):
    seen = set()
    for group in groups:
        klasses = TABLE_FUNCTION_COLLECTION.get(group, ())
        for klass in klasses:
            if klass.name not in seen:
                seen.add(klass.name)
                db.register_table_function(klass)