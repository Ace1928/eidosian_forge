import sys
from warnings import warn
import click
from .features import normalize_feature_inputs
def geojson_type_feature_opt(default=False):
    """GeoJSON Feature or Feature sequence output mode"""
    return click.option('--feature', 'geojson_type', flag_value='feature', default=default, help='Output as GeoJSON feature(s).')