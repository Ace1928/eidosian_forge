import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def shift_time(dtime, shift):
    """Adds/deletes an integer amount of seconds from a datetime specification

    :param dtime: The datatime specification
    :param shift: The wanted time shift (+/-)
    :return: A shifted datatime specification
    """
    return dtime + timedelta(seconds=shift)