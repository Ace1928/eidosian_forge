from __future__ import unicode_literals
def get_database():
    return {idstr: Lint(idstr, msgfmt, **kwargs) for idstr, msgfmt, kwargs in LINT_DB}