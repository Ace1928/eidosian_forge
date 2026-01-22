from .utils import logger, NO_VALUE
from typing import Mapping, Iterable, Callable, Union, TypeVar, Tuple, Any, List, Set, Optional, Collection, TYPE_CHECKING
def assert_config(value, options: Collection, msg='Got %r, expected one of %s'):
    if value not in options:
        raise ConfigurationError(msg % (value, options))