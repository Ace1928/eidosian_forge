import os
from typing import Dict
from ...utils.mimebundle import spec_to_mimebundle
from ..display import (
from .schema import SCHEMA_VERSION
from typing import Final
def vegalite(spec: dict, validate: bool=True) -> None:
    """Render and optionally validate a VegaLite 5 spec.

    This will use the currently enabled renderer to render the spec.

    Parameters
    ==========
    spec: dict
        A fully compliant VegaLite 5 spec, with the data portion fully processed.
    validate: bool
        Should the spec be validated against the VegaLite 5 schema?
    """
    from IPython.display import display
    display(VegaLite(spec, validate=validate))