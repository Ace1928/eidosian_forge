import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
def unregister_post_import_hook(name: str, hook_id: Optional[str]) -> None:
    with _post_import_hooks_lock:
        hooks = _post_import_hooks.get(name)
        if hooks is not None:
            if hook_id is not None:
                hooks.pop(hook_id, None)
                if not hooks:
                    del _post_import_hooks[name]
            else:
                del _post_import_hooks[name]