from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def get_j2_ctx(self, path: Union[str, Path], name: Optional[str]=None, enable_async: Optional[bool]=False, **kwargs) -> 'jinja2.Environment':
    """
        Creates a jinja2 context

        path can be relative to the module_dir or absolute
        """
    base_ctx = self.j2_actxs if enable_async else self.j2_ctxs
    name = name or self.module_name
    if name not in base_ctx:
        import jinja2
        if isinstance(path, str):
            if path.startswith('/'):
                path = Path(path)
            else:
                path = self.settings.module_path.joinpath(path)
        self.logger.info(f'Jinja2 Path: {path} for {name}')
        base_ctx[name] = jinja2.Environment(loader=jinja2.FileSystemLoader(path), enable_async=enable_async, **kwargs)
    return base_ctx[name]