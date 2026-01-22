import datetime
import re
@staticmethod
def hourly(t):
    dt = t + datetime.timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)