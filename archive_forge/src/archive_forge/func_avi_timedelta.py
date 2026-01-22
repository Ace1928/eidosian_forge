from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def avi_timedelta(td):
    """
    This is a wrapper class to workaround python 2.6 builtin datetime.timedelta
    does not have total_seconds method
    :param timedelta object
    """
    if type(td) is not timedelta:
        raise TypeError()
    if sys.version_info >= (2, 7):
        ts = td.total_seconds()
    else:
        ts = td.seconds + 24 * 3600 * td.days
    return ts