import re
from hacking import core
from neutron_lib.hacking import translation_checks
@core.flake8ext
def check_no_eventlet_imports(logical_line):
    """N535 - Usage of Python eventlet module not allowed.

    :param logical_line: The logical line to check.
    :returns: None if the logical line passes the check, otherwise a tuple
        is yielded that contains the offending index in logical line and a
        message describe the check validation failure.
    """
    if re.match('(import|from)\\s+[(]?eventlet', logical_line):
        msg = 'N535: Usage of Python eventlet module not allowed'
        yield (logical_line.index('eventlet'), msg)