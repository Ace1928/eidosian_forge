from __future__ import absolute_import, division, print_function
import errno
import os
import tempfile
def load_file_if_exists(path, module=None, ignore_errors=False):
    """
    Load the file as a bytes string. If the file does not exist, ``None`` is returned.

    If ``ignore_errors`` is ``True``, will ignore errors. Otherwise, errors are
    raised as exceptions if ``module`` is not specified, and result in ``module.fail_json``
    being called when ``module`` is specified.
    """
    try:
        with open(path, 'rb') as f:
            return f.read()
    except EnvironmentError as exc:
        if exc.errno == errno.ENOENT:
            return None
        if ignore_errors:
            return None
        if module is None:
            raise
        module.fail_json('Error while loading {0} - {1}'.format(path, str(exc)))
    except Exception as exc:
        if ignore_errors:
            return None
        if module is None:
            raise
        module.fail_json('Error while loading {0} - {1}'.format(path, str(exc)))