from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def read_aims_header_info(filename: str | Path) -> tuple[dict[str, None | list[str] | str], dict[str, Any]]:
    """Read the FHI-aims header information.

    Args:
      filename(str or Path): The file to read

    Returns:
        The metadata for the header of the aims calculation
    """
    content = None
    for path in [Path(filename), Path(f'{filename}.gz')]:
        if not path.exists():
            continue
        if path.suffix == '.gz':
            with gzip.open(filename, mode='rt') as file:
                content = file.read()
        else:
            with open(filename) as file:
                content = file.read()
    if content is None:
        raise FileNotFoundError(f'The requested output file {filename} does not exist.')
    return read_aims_header_info_from_content(content)