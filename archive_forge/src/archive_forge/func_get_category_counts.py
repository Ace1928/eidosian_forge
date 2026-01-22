from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
def get_category_counts(self):
    lint_counts = {}
    for _, file_ctx in sorted(self.file_ctxs.items()):
        for record in file_ctx.get_lint():
            category_char = record.spec.idstr[0]
            if category_char not in lint_counts:
                lint_counts[category_char] = 0
            lint_counts[category_char] += 1
    return lint_counts