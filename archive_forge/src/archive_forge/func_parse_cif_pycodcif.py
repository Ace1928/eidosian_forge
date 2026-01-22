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
def parse_cif_pycodcif(fileobj) -> Iterator[CIFBlock]:
    """Parse a CIF file using pycodcif CIF parser."""
    if not isinstance(fileobj, str):
        fileobj = fileobj.name
    try:
        from pycodcif import parse
    except ImportError:
        raise ImportError('parse_cif_pycodcif requires pycodcif ' + '(http://wiki.crystallography.net/cod-tools/pycodcif/)')
    data, _, _ = parse(fileobj)
    for datablock in data:
        tags = datablock['values']
        for tag in tags.keys():
            values = [convert_value(x) for x in tags[tag]]
            if len(values) == 1:
                tags[tag] = values[0]
            else:
                tags[tag] = values
        yield CIFBlock(datablock['name'], tags)