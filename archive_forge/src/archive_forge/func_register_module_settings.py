from __future__ import annotations
from lazyops.utils.lazy import lazy_import
from typing import Any, Dict, Optional, Type, Union, TYPE_CHECKING
def register_module_settings(module: str, settings: 'AppSettings'):
    """
    Register the module settings
    """
    _lazy_module_settings[module] = settings