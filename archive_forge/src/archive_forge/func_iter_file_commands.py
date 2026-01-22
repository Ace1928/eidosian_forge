from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def iter_file_commands(self):
    """Iterator returning FileCommand objects.

        If an invalid file command is found, the line is silently
        pushed back and iteration ends.
        """
    while True:
        line = self.next_line()
        if line is None:
            break
        elif len(line) == 0 or line.startswith(b'#'):
            continue
        elif line.startswith(b'M '):
            yield self._parse_file_modify(line[2:])
        elif line.startswith(b'D '):
            path = self._path(line[2:])
            yield commands.FileDeleteCommand(path)
        elif line.startswith(b'R '):
            old, new = self._path_pair(line[2:])
            yield commands.FileRenameCommand(old, new)
        elif line.startswith(b'C '):
            src, dest = self._path_pair(line[2:])
            yield commands.FileCopyCommand(src, dest)
        elif line.startswith(b'deleteall'):
            yield commands.FileDeleteAllCommand()
        else:
            self.push_line(line)
            break