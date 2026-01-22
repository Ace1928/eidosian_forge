import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@classmethod
def setup_completer(cls):
    """Get the dictionary of valid completions"""
    try:
        for element in Store.options().keys():
            options = Store.options()['.'.join(element)]
            plotkws = options['plot'].allowed_keywords
            stylekws = options['style'].allowed_keywords
            dotted = '.'.join(element)
            cls._completions[dotted] = (plotkws, stylekws if stylekws else [])
    except KeyError:
        pass
    return cls._completions