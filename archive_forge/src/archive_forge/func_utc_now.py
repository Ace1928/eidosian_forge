import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def utc_now():
    return calendar.timegm(time.gmtime())