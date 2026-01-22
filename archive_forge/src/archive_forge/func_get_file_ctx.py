from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
def get_file_ctx(self, infile_path, config):
    if infile_path not in self.file_ctxs:
        self.file_ctxs[infile_path] = FileContext(self, infile_path)
    ctx = self.file_ctxs[infile_path]
    ctx.config = config
    return ctx