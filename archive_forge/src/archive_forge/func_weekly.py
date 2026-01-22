import datetime
import re
@staticmethod
def weekly(t):
    dt = t + datetime.timedelta(days=7 - t.weekday())
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)