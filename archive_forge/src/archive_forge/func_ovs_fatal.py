import os
import os.path
import sys
def ovs_fatal(*args, **kwargs):
    """Prints 'message' on stderr and emits an ERROR level log message to
    'vlog' if supplied.  If 'err_no' is nonzero, then it is formatted with
    ovs_retval_to_string() and appended to the message inside parentheses.
    Then, terminates with exit code 1 (indicating a failure).

    'message' should not end with a new-line, because this function will add
    one itself."""
    ovs_error(*args, **kwargs)
    sys.exit(1)