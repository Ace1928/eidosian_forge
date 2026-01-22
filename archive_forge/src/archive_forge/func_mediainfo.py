from __future__ import division
import json
import os
import re
import sys
from subprocess import Popen, PIPE
from math import log, ceil
from tempfile import TemporaryFile
from warnings import warn
from functools import wraps
def mediainfo(filepath):
    """Return dictionary with media info(codec, duration, size, bitrate...) from filepath
    """
    prober = get_prober_name()
    command_args = ['-v', 'quiet', '-show_format', '-show_streams', filepath]
    command = [prober, '-of', 'old'] + command_args
    res = Popen(command, stdout=PIPE)
    output = res.communicate()[0].decode('utf-8')
    if res.returncode != 0:
        command = [prober] + command_args
        output = Popen(command, stdout=PIPE).communicate()[0].decode('utf-8')
    rgx = re.compile('(?:(?P<inner_dict>.*?):)?(?P<key>.*?)\\=(?P<value>.*?)$')
    info = {}
    if sys.platform == 'win32':
        output = output.replace('\r', '')
    for line in output.split('\n'):
        mobj = rgx.match(line)
        if mobj:
            inner_dict, key, value = mobj.groups()
            if inner_dict:
                try:
                    info[inner_dict]
                except KeyError:
                    info[inner_dict] = {}
                info[inner_dict][key] = value
            else:
                info[key] = value
    return info