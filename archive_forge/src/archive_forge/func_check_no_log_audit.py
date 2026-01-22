import re
from hacking import core
import pycodestyle
@core.flake8ext
def check_no_log_audit(logical_line):
    """Ensure that we are not using LOG.audit messages
    Plans are in place going forward as discussed in the following
    spec (https://review.opendev.org/#/c/132552/) to take out
    LOG.audit messages. Given that audit was a concept invented
    for OpenStack we can enforce not using it.
    """
    if 'LOG.audit(' in logical_line:
        yield (0, 'D709: LOG.audit is deprecated, please use LOG.info!')