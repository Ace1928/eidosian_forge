import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
def notify_module_loaded(module: Any) -> None:
    name = getattr(module, '__name__', None)
    with _post_import_hooks_lock:
        hooks = _post_import_hooks.pop(name, {})
    for hook in hooks.values():
        if hook:
            hook(module)