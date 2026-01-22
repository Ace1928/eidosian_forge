from __future__ import absolute_import, division
import itertools
import math
import sys
import textwrap
def load_all_palettes(names_and_lengths, get_map_func):
    """
    Load all palettes (including reversed ones) into a dictionary,
    given a dictionary of names and lengths, plus a get_map function.

    Parameters
    ----------
    names_and_lengths : dict
        names_and_lengths : sequence of tuple
        Pairs of names and lengths.
    get_map_func : function
        Function with signature get_map(name, reverse=False) that returns
        a Palette subclass instance.

    Returns
    -------
    dict
        Maps Name_Length strings to instances of a Palette subclass.

    """
    maps = {}
    for name, length in names_and_lengths:
        map_name = palette_name(name, length)
        maps[map_name] = get_map_func(map_name)
        maps[map_name + '_r'] = get_map_func(map_name, reverse=True)
    return maps