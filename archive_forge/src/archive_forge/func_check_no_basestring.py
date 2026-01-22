import re
from hacking import core
@core.flake8ext
def check_no_basestring(logical_line):
    """O343 - basestring is not Python3-compatible.

    :param logical_line: The logical line to check.
    :returns: None if the logical line passes the check, otherwise a tuple
              is yielded that contains the offending index in logical line
              and a message describe the check validation failure.
    """
    if no_basestring_re.search(logical_line):
        msg = 'O343: basestring is not Python3-compatible, use str instead.'
        yield (0, msg)