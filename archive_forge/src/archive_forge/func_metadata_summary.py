from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def metadata_summary(self) -> dict[str, list[str] | str | None]:
    """Dictionary containing all metadata for FHI-aims build."""
    return {'commit_hash': self.commit_hash, 'aims_uuid': self.aims_uuid, 'version_number': self.version_number, 'fortran_compiler': self.fortran_compiler, 'c_compiler': self.c_compiler, 'fortran_compiler_flags': self.fortran_compiler_flags, 'c_compiler_flags': self.c_compiler_flags, 'build_type': self.build_type, 'linked_against': self.linked_against}