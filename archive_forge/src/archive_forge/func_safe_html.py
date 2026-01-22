import threading
from tensorboard._vendor.bleach.sanitizer import Cleaner
import markdown
from tensorboard import context as _context
from tensorboard.backend import experiment_id as _experiment_id
from tensorboard.util import tb_logging
def safe_html(unsafe_string):
    """Return the input as a str, sanitized for insertion into the DOM.

    Arguments:
      unsafe_string: A Unicode string or UTF-8--encoded bytestring
        possibly containing unsafe HTML markup.

    Returns:
      A string containing safe HTML.
    """
    total_null_bytes = 0
    if isinstance(unsafe_string, bytes):
        unsafe_string = unsafe_string.decode('utf-8')
    return _CLEANER_STORE.cleaner.clean(unsafe_string)