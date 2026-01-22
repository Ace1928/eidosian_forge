from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def format_date_time(timestamp):
    year, month, day, hh, mm, ss, wd, y, z = time.gmtime(timestamp)
    return '%s, %02d %3s %4d %02d:%02d:%02d GMT' % (_weekdayname[wd], day, _monthname[month], year, hh, mm, ss)