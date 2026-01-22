from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
def is_idstr(self, idstr):
    return idstr in self.global_ctx.lintdb