import io
import re
import shlex
import warnings
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any, Sequence
import collections.abc
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.io.cif_unicode import format_unicode, handle_subscripts
from ase.utils import iofunction
@iofunction('wb')
def write_cif(fd, images, cif_format=None, wrap=True, labels=None, loop_keys=None) -> None:
    """Write *images* to CIF file.

    wrap: bool
        Wrap atoms into unit cell.

    labels: list
        Use this list (shaped list[i_frame][i_atom] = string) for the
        '_atom_site_label' section instead of automatically generating
        it from the element symbol.

    loop_keys: dict
        Add the information from this dictionary to the `loop_`
        section.  Keys are printed to the `loop_` section preceeded by
        ' _'. dict[key] should contain the data printed for each atom,
        so it needs to have the setup `dict[key][i_frame][i_atom] =
        string`. The strings are printed as they are, so take care of
        formating. Information can be re-read using the `store_tags`
        option of the cif reader.

    """
    if cif_format is not None:
        warnings.warn('The cif_format argument is deprecated and may be removed in the future.  Use loop_keys to customize data written in loop.', FutureWarning)
    if loop_keys is None:
        loop_keys = {}
    if hasattr(images, 'get_positions'):
        images = [images]
    fd = io.TextIOWrapper(fd, encoding='latin-1')
    try:
        for i, atoms in enumerate(images):
            blockname = f'data_image{i}\n'
            image_loop_keys = {key: loop_keys[key][i] for key in loop_keys}
            write_cif_image(blockname, atoms, fd, wrap=wrap, labels=None if labels is None else labels[i], loop_keys=image_loop_keys)
    finally:
        fd.detach()