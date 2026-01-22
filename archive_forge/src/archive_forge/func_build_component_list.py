import numbers
import os
import sys
import warnings
from configparser import ConfigParser
from operator import itemgetter
from pathlib import Path
from typing import (
from scrapy.exceptions import ScrapyDeprecationWarning, UsageError
from scrapy.settings import BaseSettings
from scrapy.utils.deprecate import update_classpath
from scrapy.utils.python import without_none_values
def build_component_list(compdict: MutableMapping[Any, Any], custom: Any=None, convert: Callable[[Any], Any]=update_classpath) -> List[Any]:
    """Compose a component list from a { class: order } dictionary."""

    def _check_components(complist: Collection[Any]) -> None:
        if len({convert(c) for c in complist}) != len(complist):
            raise ValueError(f'Some paths in {complist!r} convert to the same object, please update your settings')

    def _map_keys(compdict: Mapping[Any, Any]) -> Union[BaseSettings, Dict[Any, Any]]:
        if isinstance(compdict, BaseSettings):
            compbs = BaseSettings()
            for k, v in compdict.items():
                prio = compdict.getpriority(k)
                assert prio is not None
                if compbs.getpriority(convert(k)) == prio:
                    raise ValueError(f'Some paths in {list(compdict.keys())!r} convert to the same object, please update your settings')
                else:
                    compbs.set(convert(k), v, priority=prio)
            return compbs
        _check_components(compdict)
        return {convert(k): v for k, v in compdict.items()}

    def _validate_values(compdict: Mapping[Any, Any]) -> None:
        """Fail if a value in the components dict is not a real number or None."""
        for name, value in compdict.items():
            if value is not None and (not isinstance(value, numbers.Real)):
                raise ValueError(f'Invalid value {value} for component {name}, please provide a real number or None instead')
    if custom is not None:
        warnings.warn("The 'custom' attribute of build_component_list() is deprecated. Please merge its value into 'compdict' manually or change your code to use Settings.getwithbase().", category=ScrapyDeprecationWarning, stacklevel=2)
        if isinstance(custom, (list, tuple)):
            _check_components(custom)
            return type(custom)((convert(c) for c in custom))
        compdict.update(custom)
    _validate_values(compdict)
    compdict = without_none_values(_map_keys(compdict))
    return [k for k, v in sorted(compdict.items(), key=itemgetter(1))]