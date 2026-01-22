from .error import *
from .tokens import *
from .events import *
from .nodes import *
from .loader import *
from .dumper import *
import io
def load_warning(method):
    if _warnings_enabled['YAMLLoadWarning'] is False:
        return
    import warnings
    message = 'calling yaml.%s() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.' % method
    warnings.warn(message, YAMLLoadWarning, stacklevel=3)