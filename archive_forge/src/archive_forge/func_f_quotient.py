import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def f_quotient(arg0, arg1, arg2=0):
    if arg2:
        return int((arg0 - arg1) / (arg2 - arg1))
    elif not arg0:
        return 0
    else:
        return int(arg0 / arg1)