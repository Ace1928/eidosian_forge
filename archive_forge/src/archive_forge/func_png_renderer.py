import os
from typing import Dict
from ...utils.mimebundle import spec_to_mimebundle
from ..display import (
from .schema import SCHEMA_VERSION
from typing import Final
def png_renderer(spec: dict, **metadata) -> Dict[str, bytes]:
    return spec_to_mimebundle(spec, format='png', mode='vega-lite', vega_version=VEGA_VERSION, vegaembed_version=VEGAEMBED_VERSION, vegalite_version=VEGALITE_VERSION, **metadata)