from __future__ import annotations
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core import Lattice, Structure
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import CompletePhononDos, PhononDos
from pymatgen.phonon.gruneisen import GruneisenParameter, GruneisenPhononBandStructureSymmLine
from pymatgen.phonon.thermal_displacements import ThermalDisplacementMatrices
from pymatgen.symmetry.bandstructure import HighSymmKpath
def get_ph_bs_symm_line_from_dict(bands_dict, has_nac=False, labels_dict=None):
    """
    Creates a pymatgen PhononBandStructure object from the dictionary
    extracted by the band.yaml file produced by phonopy. The labels
    will be extracted from the dictionary, if present. If the 'eigenvector'
    key is found the eigendisplacements will be calculated according to the
    formula:

        exp(2*pi*i*(frac_coords \\\\dot q) / sqrt(mass) * v

    and added to the object.

    Args:
        bands_dict: the dictionary extracted from the band.yaml file
        has_nac: True if the data have been obtained with the option
            --nac option. Default False.
        labels_dict: dict that links a qpoint in frac coords to a label.
            Its value will replace the data contained in the band.yaml.
    """
    structure = get_structure_from_dict(bands_dict)
    q_pts = []
    frequencies = []
    eigen_displacements = []
    phonopy_labels_dict = {}
    for phonon in bands_dict['phonon']:
        q_pos = phonon['q-position']
        q_pts.append(q_pos)
        bands = []
        eig_q = []
        for band in phonon['band']:
            bands.append(band['frequency'])
            if 'eigenvector' in band:
                eig_b = []
                for idx, eig_a in enumerate(band['eigenvector']):
                    eig_vec = np.zeros(3, complex)
                    for x in range(3):
                        eig_vec[x] = eig_a[x][0] + eig_a[x][1] * 1j
                    eig_b.append(eigvec_to_eigdispl(eig_vec, q_pos, structure[idx].frac_coords, structure.site_properties['phonopy_masses'][idx]))
                eig_q.append(eig_b)
        frequencies.append(bands)
        if 'label' in phonon:
            phonopy_labels_dict[phonon['label']] = phonon['q-position']
        if eig_q:
            eigen_displacements.append(eig_q)
    q_pts = np.array(q_pts)
    frequencies = np.transpose(frequencies)
    if eigen_displacements:
        eigen_displacements = np.transpose(eigen_displacements, (1, 0, 2, 3))
    rec_lattice = Lattice(bands_dict['reciprocal_lattice'])
    labels_dict = labels_dict or phonopy_labels_dict
    return PhononBandStructureSymmLine(q_pts, frequencies, rec_lattice, has_nac=has_nac, labels_dict=labels_dict, structure=structure, eigendisplacements=eigen_displacements)