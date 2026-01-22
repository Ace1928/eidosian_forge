import os
import re
from os.path import sep
from os.path import join as slash  # just like that name better
from os.path import dirname, abspath
import kivy
from kivy.logger import Logger
import textwrap
def iter_filename_info(dir_name):
    """
    Yield info (dict) of each matching screenshot found walking the
    directory dir_name. A matching screenshot uses double underscores to
    separate fields, i.e. path__to__filename__py.png as the screenshot for
    examples/path/to/filename.py.

    Files not ending with .png are ignored, others are either parsed or
    yield an error.

    Info fields 'dunder', 'dir', 'file', 'ext', 'source' if not 'error'
    """
    pattern = re.compile('^((.+)__(.+)__([^-]+))\\.png')
    for t in os.walk(dir_name):
        for filename in t[2]:
            if filename.endswith('.png'):
                m = pattern.match(filename)
                if m is None:
                    yield {'error': 'png filename not following screenshot pattern: {}'.format(filename)}
                else:
                    d = m.group(2).replace('__', sep)
                    yield {'dunder': m.group(1), 'dir': d, 'file': m.group(3), 'ext': m.group(4), 'source': slash(d, m.group(3) + '.' + m.group(4))}