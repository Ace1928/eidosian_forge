import threading
from tensorboard._vendor.bleach.sanitizer import Cleaner
import markdown
from tensorboard import context as _context
from tensorboard.backend import experiment_id as _experiment_id
from tensorboard.util import tb_logging
def markdown_to_safe_html(markdown_string):
    """Convert Markdown to HTML that's safe to splice into the DOM.

    Arguments:
      markdown_string: A Unicode string or UTF-8--encoded bytestring
        containing Markdown source. Markdown tables are supported.

    Returns:
      A string containing safe HTML.
    """
    return markdowns_to_safe_html([markdown_string], lambda xs: xs[0])