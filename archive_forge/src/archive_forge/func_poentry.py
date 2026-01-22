import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
def poentry(self, path, lineno, s, comment=None):
    if self._msgids is not None:
        if s in self._msgids:
            return
        self._msgids.add(s)
    if comment is None:
        comment = ''
    else:
        comment = '# %s\n' % comment
    mutter('Exporting msg %r at line %d in %r', s[:20], lineno, path)
    line = '#: {path}:{lineno}\n{comment}msgid {msg}\nmsgstr ""\n\n'.format(path=path, lineno=lineno, comment=comment, msg=_normalize(s))
    self.outf.write(line)