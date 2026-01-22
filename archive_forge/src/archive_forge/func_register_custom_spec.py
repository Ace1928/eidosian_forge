import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@classmethod
def register_custom_spec(cls, spec):
    spec, _ = StoreOptions.expand_compositor_keys(spec)
    errmsg = StoreOptions.validation_error_message(spec)
    if errmsg:
        cls.error_message = errmsg
    cls.opts_spec = spec