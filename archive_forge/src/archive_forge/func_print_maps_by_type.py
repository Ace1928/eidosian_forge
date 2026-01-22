from __future__ import absolute_import
from __future__ import print_function
import webbrowser
from ..palette import Palette
from .colorbrewer_all_schemes import COLOR_MAPS
def print_maps_by_type(map_type, number=None):
    """
    Print all available maps of a given type.

    Parameters
    ----------
    map_type : {'sequential', 'diverging', 'qualitative'}
        Select map type to print.
    number : int, optional
        Filter output by number of defined colors. By default there is
        no numeric filtering.

    """
    if map_type.lower() not in MAP_TYPES:
        s = 'Invalid map type, must be one of {0}'.format(MAP_TYPES)
        raise ValueError(s)
    print(map_type)
    map_type = map_type.capitalize()
    map_keys = sorted(COLOR_MAPS[map_type].keys())
    format_str = '{0:8}  :  {1}'
    for mk in map_keys:
        num_keys = sorted(COLOR_MAPS[map_type][mk].keys(), key=int)
        if not number or str(number) in num_keys:
            num_str = '{' + ', '.join(num_keys) + '}'
            print(format_str.format(mk, num_str))