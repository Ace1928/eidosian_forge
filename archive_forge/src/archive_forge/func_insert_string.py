import functools
import re
import sys
from Xlib.support import lock
def insert_string(self, data):
    """insert_string(data)

        Insert the resources entries in the string DATA into the
        database.

        """
    lines = data.split('\n')
    while lines:
        line = lines[0]
        del lines[0]
        if not line:
            continue
        if comment_re.match(line):
            continue
        while line[-1] == '\\':
            if lines:
                line = line[:-1] + lines[0]
                del lines[0]
            else:
                line = line[:-1]
                break
        m = resource_spec_re.match(line)
        if not m:
            continue
        res, value = m.group(1, 2)
        splits = value_escape_re.split(value)
        for i in range(1, len(splits), 2):
            s = splits[i]
            if len(s) == 3:
                splits[i] = chr(int(s, 8))
            elif s == 'n':
                splits[i] = '\n'
        splits[-1] = splits[-1].rstrip()
        value = ''.join(splits)
        self.insert(res, value)