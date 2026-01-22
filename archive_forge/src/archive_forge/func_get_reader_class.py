import sys
from docutils import utils, parsers, Component
from docutils.transforms import universal
def get_reader_class(reader_name):
    """Return the Reader class from the `reader_name` module."""
    reader_name = reader_name.lower()
    if reader_name in _reader_aliases:
        reader_name = _reader_aliases[reader_name]
    try:
        module = __import__(reader_name, globals(), locals(), level=1)
    except ImportError:
        module = __import__(reader_name, globals(), locals(), level=0)
    return module.Reader