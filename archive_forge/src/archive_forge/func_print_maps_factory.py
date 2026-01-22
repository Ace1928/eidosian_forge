from __future__ import absolute_import, division
import itertools
import math
import sys
import textwrap
def print_maps_factory(desc, names_and_lengths, palette_type):
    """
    Create a function that will print the names and lengths of palettes.

    Parameters
    ----------
    desc : str
        Short description of palettes, for example "sequential cmocean".
        Used to populate the print_maps docstring.
    names_and_lengths : sequence of tuple
        Pairs of names and lengths.
    palette_type : str
        Palette type to include in printed messages.

    Returns
    -------
    function
        Takes no arguments.

    """

    def print_maps():
        namelen = max((len(palette_name(name, length)) for name, length in names_and_lengths))
        fmt = '{0:' + str(namelen + 4) + '}{1:16}{2:}'
        for name, length in names_and_lengths:
            print(fmt.format(palette_name(name, length), palette_type, length))
    print_maps.__doc__ = 'Print a list of {0} palettes'.format(desc)
    return print_maps