import sys
from . import delta as _mod_delta
from . import errors as errors
from . import hooks as _mod_hooks
from . import log, osutils
from . import revision as _mod_revision
from . import tsort
from .trace import mutter, warning
from .workingtree import ShelvingUnsupported
def show_log_message(rev, prefix):
    if term_width is None:
        width = term_width
    else:
        width = term_width - len(prefix)
    log_message = log_formatter.log_string(None, rev, width, prefix=prefix)
    to_file.write(log_message + '\n')