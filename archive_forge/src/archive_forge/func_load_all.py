from .error import *
from .tokens import *
from .events import *
from .nodes import *
from .loader import *
from .dumper import *
import io
def load_all(stream, Loader=None):
    """
    Parse all YAML documents in a stream
    and produce corresponding Python objects.
    """
    if Loader is None:
        load_warning('load_all')
        Loader = FullLoader
    loader = Loader(stream)
    try:
        while loader.check_data():
            yield loader.get_data()
    finally:
        loader.dispose()