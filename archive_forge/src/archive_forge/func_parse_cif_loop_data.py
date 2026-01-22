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
def parse_cif_loop_data(lines: List[str], ncolumns: int) -> List[List[CIFDataValue]]:
    columns: List[List[CIFDataValue]] = [[] for _ in range(ncolumns)]
    tokens = []
    while lines:
        line = lines.pop().strip()
        lowerline = line.lower()
        if not line or line.startswith('_') or lowerline.startswith('data_') or lowerline.startswith('loop_'):
            lines.append(line)
            break
        if line.startswith('#'):
            continue
        if line.startswith(';'):
            moretokens = [parse_multiline_string(lines, line)]
        elif ncolumns == 1:
            moretokens = [line]
        else:
            moretokens = shlex.split(line, posix=False)
        tokens += moretokens
        if len(tokens) < ncolumns:
            continue
        if len(tokens) == ncolumns:
            for i, token in enumerate(tokens):
                columns[i].append(convert_value(token))
        else:
            warnings.warn('Wrong number {} of tokens, expected {}: {}'.format(len(tokens), ncolumns, tokens))
        tokens = []
    if tokens:
        assert len(tokens) < ncolumns
        raise RuntimeError('CIF loop ended unexpectedly with incomplete row')
    return columns