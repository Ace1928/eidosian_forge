from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
@staticmethod
def get_dos_fp_similarity(fp1: NamedTuple, fp2: NamedTuple, col: int=1, pt: int | str='All', normalize: bool=False, tanimoto: bool=False) -> float:
    """Calculates the similarity index (dot product) of two fingerprints.

        Args:
            fp1 (NamedTuple): The 1st dos fingerprint object
            fp2 (NamedTuple): The 2nd dos fingerprint object
            col (int): The item in the fingerprints (0:energies,1: densities) to take the dot product of (default is 1)
            pt (int or str) : The index of the point that the dot product is to be taken (default is All)
            normalize (bool): If True normalize the scalar product to 1 (default is False)
            tanimoto (bool): If True will compute Tanimoto index (default is False)

        Raises:
            ValueError: If both tanimoto and normalize are set to True.

        Returns:
            float: Similarity index given by the dot product
        """
    fp1_dict = CompleteDos.fp_to_dict(fp1) if not isinstance(fp1, dict) else fp1
    fp2_dict = CompleteDos.fp_to_dict(fp2) if not isinstance(fp2, dict) else fp2
    if pt == 'All':
        vec1 = np.array([pt[col] for pt in fp1_dict.values()]).flatten()
        vec2 = np.array([pt[col] for pt in fp2_dict.values()]).flatten()
    else:
        vec1 = fp1_dict[fp1[2][pt]][col]
        vec2 = fp2_dict[fp2[2][pt]][col]
    if not normalize and tanimoto:
        rescale = np.linalg.norm(vec1) ** 2 + np.linalg.norm(vec2) ** 2 - np.dot(vec1, vec2)
        return np.dot(vec1, vec2) / rescale
    if not tanimoto and normalize:
        rescale = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return np.dot(vec1, vec2) / rescale
    if not tanimoto and (not normalize):
        rescale = 1.0
        return np.dot(vec1, vec2) / rescale
    raise ValueError('Cannot compute similarity index. Please set either normalize=True or tanimoto=True or both to False.')