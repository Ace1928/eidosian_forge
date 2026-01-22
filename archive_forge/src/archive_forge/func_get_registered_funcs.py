from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import typer
from thinc.api import Config
from thinc.config import VARIABLE_RE
from wasabi import msg, table
from .. import util
from ..schemas import ConfigSchemaInit, ConfigSchemaTraining
from ..util import registry
from ._util import (
def get_registered_funcs(config: Config) -> List[Dict[str, Optional[Union[str, int]]]]:
    result = []
    for key, value in util.walk_dict(config):
        if not key[-1].startswith('@'):
            continue
        reg_name = key[-1][1:]
        registry = getattr(util.registry, reg_name)
        path = '.'.join(key[:-1])
        info = registry.find(value)
        result.append({'name': value, 'registry': reg_name, 'path': path, **info})
    return result