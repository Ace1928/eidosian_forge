from __future__ import absolute_import, division
import itertools
import math
import sys
import textwrap
def get_map_factory(desc, mod_path, names_to_data, palette_type, palette_class, is_evenly_spaced=True):
    """
    Create a function that builds a Palette instance from available data
    and the given Palette sub-class.

    Parameters
    ----------
    desc : str
        Short description of palettes, for example "sequential cmocean".
        Used to populate the get_map docstring.
    mod_path : str
        Path to module where get_map will be used. Use to point users to
        the print_maps function. E.g. 'palettable.cmocean.sequential'.
    names_to_data : dict
        Dictionary mapping string palette names to color data.
        (Lists of 0-255 integer RGB tuples.)
    palette_type : str
        Palette type to pass into Palette subclass, e.g. 'diverging'.
    palette_class : Palette subclass
        Subclass of Palette to use when creating new Palettes.
        Must have __init__ signature (name, palette_type, colors).
    is_evenly_spaced : bool
        Sets sampling of palette. If True, then choose values evenly spaced
        in palette array, otherwise choose colors from the beginning of the
        array.

    Returns
    -------
    function
        The function will have the definition:

            def get_map(name, reverse=False):

    """
    name_map = make_name_map(names_to_data.keys())

    def get_map(name, reverse=False):
        name, length = split_name_length(name)
        name_lower = name.lower()
        if name_lower not in name_map:
            raise KeyError('Unknown palette name: {0}'.format(name))
        name = name_map[name_lower]
        if len(names_to_data[name]) < length:
            raise ValueError('Number of requested colors larger than the number available in {0}'.format(name))
        if is_evenly_spaced:
            colors = evenly_spaced_values(length, names_to_data[name])
        else:
            colors = names_to_data[name][:length]
        name = palette_name(name, length)
        if reverse:
            name += '_r'
            colors = list(reversed(colors))
        return palette_class(name, palette_type, colors)
    get_map.__doc__ = textwrap.dedent('        Get a {0} palette by name.\n\n        Parameters\n        ----------\n        name : str\n            Name of map. Use {1}.print_maps\n            to see available names.\n        reverse : bool, optional\n            If True reverse colors from their default order.\n\n        Returns\n        -------\n        {2}\n\n        ').format(desc, mod_path, palette_class.__class__.__name__)
    return get_map