import sys
from warnings import warn
import click
from .features import normalize_feature_inputs
def geojson_type_bbox_opt(default=False):
    """GeoJSON bbox output mode"""
    return click.option('--bbox', 'geojson_type', flag_value='bbox', default=default, help='Output as GeoJSON bounding box array(s).')