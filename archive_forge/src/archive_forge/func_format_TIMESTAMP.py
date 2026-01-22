from time import localtime
from datetime import date, datetime, time, timedelta
from MySQLdb._mysql import string_literal
def format_TIMESTAMP(d):
    """
    :type d: datetime.datetime
    """
    if d.microsecond:
        fmt = ' '.join(['{0.year:04}-{0.month:02}-{0.day:02}', '{0.hour:02}:{0.minute:02}:{0.second:02}.{0.microsecond:06}'])
    else:
        fmt = ' '.join(['{0.year:04}-{0.month:02}-{0.day:02}', '{0.hour:02}:{0.minute:02}:{0.second:02}'])
    return fmt.format(d)