import re
import tokenize
from hacking import core
import pycodestyle
@core.flake8ext
def import_db_only_in_conductor(logical_line, filename):
    """Check that db calls are only in conductor module and in tests.

    S361
    """
    if _any_in(filename, 'sahara/conductor', 'sahara/tests', 'sahara/db'):
        return
    if _starts_with_any(logical_line, 'from sahara import db', 'from sahara.db', 'import sahara.db'):
        yield (0, 'S361: sahara.db import only allowed in sahara/conductor/*')