import threading
from tensorboard._vendor.bleach.sanitizer import Cleaner
import markdown
from tensorboard import context as _context
from tensorboard.backend import experiment_id as _experiment_id
from tensorboard.util import tb_logging
def markdowns_to_safe_html(markdown_strings, combine):
    """Convert multiple Markdown documents to one safe HTML document.

    One could also achieve this by calling `markdown_to_safe_html`
    multiple times and combining the results. Compared to that approach,
    this function may be faster, because HTML sanitization (which can be
    expensive) is performed only once rather than once per input. It may
    also be less precise: if one of the input documents has unsafe HTML
    that is sanitized away, that sanitization might affect other
    documents, even if those documents are safe.

    Args:
      markdown_strings: List of Markdown source strings to convert, as
        Unicode strings or UTF-8--encoded bytestrings. Markdown tables
        are supported.
      combine: Callback function that takes a list of unsafe HTML
        strings of the same shape as `markdown_strings` and combines
        them into a single unsafe HTML string, which will be sanitized
        and returned.

    Returns:
      A string containing safe HTML.
    """
    unsafe_htmls = []
    total_null_bytes = 0
    for source in markdown_strings:
        if isinstance(source, bytes):
            source_decoded = source.decode('utf-8')
            source = source_decoded.replace('\x00', '')
            total_null_bytes += len(source_decoded) - len(source)
        unsafe_html = _MARKDOWN_STORE.markdown.convert(source)
        unsafe_htmls.append(unsafe_html)
    unsafe_combined = combine(unsafe_htmls)
    sanitized_combined = _CLEANER_STORE.cleaner.clean(unsafe_combined)
    warning = ''
    if total_null_bytes:
        warning = '<!-- WARNING: discarded %d null bytes in markdown string after UTF-8 decoding -->\n' % total_null_bytes
    return warning + sanitized_combined