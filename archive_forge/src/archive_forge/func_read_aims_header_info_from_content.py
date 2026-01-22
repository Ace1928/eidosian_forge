from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def read_aims_header_info_from_content(content: str) -> tuple[dict[str, list[str] | None | str], dict[str, Any]]:
    """Read the FHI-aims header information.

    Args:
      content (str): The content of the output file to check

    Returns:
        The metadata for the header of the aims calculation
    """
    header_chunk = get_header_chunk(content)
    return (header_chunk.metadata_summary, header_chunk.header_summary)