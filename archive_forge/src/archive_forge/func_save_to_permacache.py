import os
import pickle
import re
import requests
import sys
import time
from datetime import datetime
from functools import wraps
from tempfile import gettempdir
def save_to_permacache():
    """Save the in-memory cache data to the permacache.

        There is a race condition here between two processes updating at the
        same time. It's perfectly acceptable to lose and/or corrupt the
        permacache information as each process's in-memory cache will remain
        in-tact.

        """
    update_from_permacache()
    try:
        with open(filename, 'wb') as fp:
            pickle.dump(cache, fp, pickle.HIGHEST_PROTOCOL)
    except IOError:
        pass