from ...utils._importers import import_vl_convert
from ...utils.compiler import VegaLiteCompilerRegistry
from typing import Final
def vl_convert_compiler(vegalite_spec: dict) -> dict:
    """
    Vega-Lite to Vega compiler that uses vl-convert
    """
    from . import SCHEMA_VERSION
    vlc = import_vl_convert()
    vl_version = '_'.join(SCHEMA_VERSION.split('.')[:2])
    return vlc.vegalite_to_vega(vegalite_spec, vl_version=vl_version)