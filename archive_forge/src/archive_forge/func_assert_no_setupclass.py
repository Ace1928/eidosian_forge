import re
from hacking import core
@core.flake8ext
def assert_no_setupclass(logical_line):
    """Check for use of setUpClass

    O300
    """
    if SETUPCLASS_RE.match(logical_line):
        yield (0, 'O300: setUpClass not allowed')