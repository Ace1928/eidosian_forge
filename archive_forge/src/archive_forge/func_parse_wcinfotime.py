import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def parse_wcinfotime(timestr):
    """ Returns seconds since epoch, UTC. """
    m = re.match('(\\d+-\\d+-\\d+ \\d+:\\d+:\\d+) ([+-]\\d+) .*', timestr)
    if not m:
        raise ValueError('timestring %r does not match' % timestr)
    timestr, timezone = m.groups()
    parsedtime = time.strptime(timestr, '%Y-%m-%d %H:%M:%S')
    return calendar.timegm(parsedtime)