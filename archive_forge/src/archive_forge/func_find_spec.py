import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
def find_spec(self, fullname: str, path: Optional[str]=None, target: Any=None) -> Any:
    with _post_import_hooks_lock:
        if fullname not in _post_import_hooks:
            return None
    if fullname in self.in_progress:
        return None
    self.in_progress[fullname] = True
    try:
        spec = find_spec(fullname)
        loader = getattr(spec, 'loader', None)
        if loader and (not isinstance(loader, _ImportHookChainedLoader)):
            assert spec is not None
            spec.loader = _ImportHookChainedLoader(loader)
        return spec
    finally:
        del self.in_progress[fullname]