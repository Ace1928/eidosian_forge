import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name  # type: ignore
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.utils import is_valid_field
def parse_toml(config_file: str) -> Optional[Dict[str, Any]]:
    if not config_file.endswith('.toml'):
        return None
    read_mode = 'rb'
    if sys.version_info >= (3, 11):
        import tomllib as toml_
    else:
        try:
            import tomli as toml_
        except ImportError:
            read_mode = 'r'
            try:
                import toml as toml_
            except ImportError:
                import warnings
                warnings.warn('No TOML parser installed, cannot read configuration from `pyproject.toml`.')
                return None
    with open(config_file, read_mode) as rf:
        return toml_.load(rf)