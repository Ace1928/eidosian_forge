import functools
import re
import sys
from Xlib.support import lock
def copy_db(db):
    newdb = {}
    for comp, group in db.items():
        newdb[comp] = copy_group(group)
    return newdb