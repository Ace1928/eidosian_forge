from itertools import chain
import json
import re
import click
def normalize_feature_inputs(ctx, param, value):
    """Click callback that normalizes feature input values.

    Returns a generator over features from the input value.

    Parameters
    ----------
    ctx: a Click context
    param: the name of the argument or option
    value: object
        The value argument may be one of the following:

        1. A list of paths to files containing GeoJSON feature
           collections or feature sequences.
        2. A list of string-encoded coordinate pairs of the form
           "[lng, lat]", or "lng, lat", or "lng lat".

        If no value is provided, features will be read from stdin.

    Yields
    ------
    Mapping
        A GeoJSON Feature represented by a Python mapping

    """
    for feature_like in value or ('-',):
        try:
            with click.open_file(feature_like, encoding='utf-8') as src:
                for feature in iter_features(iter(src)):
                    yield feature
        except IOError:
            coords = list(coords_from_query(feature_like))
            yield {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Point', 'coordinates': coords}}