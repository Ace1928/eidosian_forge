import os
import pickle
import re
import requests
import sys
import time
from datetime import datetime
from functools import wraps
from tempfile import gettempdir
def update_from_permacache():
    """Attempt to update newer items from the permacache."""
    try:
        with open(filename, 'rb') as fp:
            permacache = pickle.load(fp)
    except Exception:
        return
    for key, value in permacache.items():
        if key not in cache or value[0] > cache[key][0]:
            cache[key] = value