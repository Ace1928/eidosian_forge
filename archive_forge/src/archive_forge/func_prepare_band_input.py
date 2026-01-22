from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.symmetry.bandstructure import HighSymmKpath
def prepare_band_input(structure: Structure, density: float=20):
    """Prepare the band information needed for the FHI-aims control.in file.

    Parameters
    ----------
    structure: Structure
        The structure for which the band path is calculated
    density: float
        Number of kpoints per Angstrom.
    """
    bp = HighSymmKpath(structure)
    points, labels = bp.get_kpoints(line_density=density, coords_are_cartesian=False)
    lines_and_labels: list[_SegmentDict] = []
    current_segment: _SegmentDict | None = None
    for label_, coords in zip(labels, points):
        label = 'G' if label_ in ('GAMMA', '\\Gamma', 'Γ') else label_
        if label:
            if current_segment is None:
                current_segment = _SegmentDict(coords=[coords], labels=[label], length=1)
            else:
                current_segment['coords'].append(coords)
                current_segment['labels'].append(label)
                current_segment['length'] += 1
                lines_and_labels.append(current_segment)
                current_segment = None
        elif current_segment is not None:
            current_segment['length'] += 1
    bands = []
    for segment in lines_and_labels:
        start, end = segment['coords']
        label_start, label_end = segment['labels']
        bands.append(f'band {start[0]:9.5f}{start[1]:9.5f}{start[2]:9.5f} {end[0]:9.5f}{end[1]:9.5f}{end[2]:9.5f} {segment['length']:4d} {label_start:3}{label_end:3}')
    return bands