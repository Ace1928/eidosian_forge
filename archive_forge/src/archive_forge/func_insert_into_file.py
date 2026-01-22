import pkg_resources
import sys
import optparse
from . import bool_optparse
import os
import re
import textwrap
from . import pluginlib
import configparser
import getpass
from logging.config import fileConfig
def insert_into_file(self, filename, marker_name, text, indent=False):
    """
        Inserts ``text`` into the file, right after the given marker.
        Markers look like: ``-*- <marker_name>[:]? -*-``, and the text
        will go on the immediately following line.

        Raises ``ValueError`` if the marker is not found.

        If ``indent`` is true, then the text will be indented at the
        same level as the marker.
        """
    if not text.endswith('\n'):
        raise ValueError('The text must end with a newline: %r' % text)
    if not os.path.exists(filename) and self.simulate:
        if self.verbose:
            print('Would (if not simulating) insert text into %s' % self.shorten(filename))
        return
    f = open(filename)
    lines = f.readlines()
    f.close()
    regex = re.compile('-\\*-\\s+%s:?\\s+-\\*-' % re.escape(marker_name), re.I)
    for i in range(len(lines)):
        if regex.search(lines[i]):
            if lines[i:] and len(lines[i:]) > 1 and ''.join(lines[i + 1:]).strip().startswith(text.strip()):
                print('Warning: line already found in %s (not inserting' % filename)
                print('  %s' % lines[i])
                return
            if indent:
                text = text.lstrip()
                match = re.search('^[ \\t]*', lines[i])
                text = match.group(0) + text
            lines[i + 1:i + 1] = [text]
            break
    else:
        errstr = "Marker '-*- %s -*-' not found in %s" % (marker_name, filename)
        if 1 or self.simulate:
            print('Warning: %s' % errstr)
        else:
            raise ValueError(errstr)
    if self.verbose:
        print('Updating %s' % self.shorten(filename))
    if not self.simulate:
        f = open(filename, 'w')
        f.write(''.join(lines))
        f.close()