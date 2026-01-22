import re
from hacking import core
@core.flake8ext
def check_python3_no_iteritems(logical_line):
    msg = 'HE302: Use dict.items() instead of dict.iteritems().'
    if re.search('.*\\.iteritems\\(\\)', logical_line):
        yield (0, msg)