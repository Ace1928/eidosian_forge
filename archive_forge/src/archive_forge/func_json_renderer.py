import os
from typing import Dict
from ...utils.mimebundle import spec_to_mimebundle
from ..display import (
from .schema import SCHEMA_VERSION
from typing import Final
def json_renderer(spec: dict, **metadata) -> DefaultRendererReturnType:
    return json_renderer_base(spec, DEFAULT_DISPLAY, **metadata)