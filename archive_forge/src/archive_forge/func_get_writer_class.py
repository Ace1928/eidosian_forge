import os.path
import sys
import docutils
from docutils import languages, Component
from docutils.transforms import universal
def get_writer_class(writer_name):
    """Return the Writer class from the `writer_name` module."""
    writer_name = writer_name.lower()
    if writer_name in _writer_aliases:
        writer_name = _writer_aliases[writer_name]
    try:
        module = __import__(writer_name, globals(), locals(), level=1)
    except ImportError:
        module = __import__(writer_name, globals(), locals(), level=0)
    return module.Writer