import threading
import warnings
from oslo_utils import importutils
from oslo_utils import timeutils
def is_monkey_patched(module):
    """Determines safely is eventlet patching for module enabled or not
    :param module: String, module name
    :return Bool: True if module is patched, False otherwise
    """
    if _patcher is None:
        return False
    return _patcher.is_monkey_patched(module)