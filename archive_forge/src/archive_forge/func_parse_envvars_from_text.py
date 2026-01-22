from __future__ import annotations
import os
import re
import json
from pathlib import Path
from lazyops.utils.helpers import build_dict_from_str
from typing import Any, Dict, List, Optional, Tuple, Union, Type, TypeVar, TYPE_CHECKING
def parse_envvars_from_text(text: str, values: Optional[Dict[str, Any]]=None, envvar_prefix: Optional[str]='env/') -> Tuple[str, Dict[str, Any]]:
    """
    Parse values from a text block

    Returns:
        Tuple[str, Dict[str, Any]]: The text with the envvars replaced and the parsed values
    """
    _prefix = envvar_prefix.replace('/', '\\/')
    pattern = re.compile(f'({_prefix}\\w+)')
    values = values or {}
    matches = pattern.findall(text)
    for match in matches:
        envvar = match.replace(envvar_prefix, '')
        _default = values.get(envvar)
        if isinstance(_default, type):
            _type = _default
            _default = None
        else:
            _type = type(_default) if _default is not None else None
        val = parse_from_envvar(envvar, default=_default, _type=_type)
        values[envvar] = val
        text = text.replace(match, val if val is not None else '')
    return (text, values)