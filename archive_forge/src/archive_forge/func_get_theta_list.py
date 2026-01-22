import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def get_theta_list(self):
    """List of theta angles for all 3 consecutive Calpha atoms."""
    theta_list = []
    ca_list = self.get_ca_list()
    for i in range(len(ca_list) - 2):
        atom_list = (ca_list[i], ca_list[i + 1], ca_list[i + 2])
        v1, v2, v3 = (a.get_vector() for a in atom_list)
        theta = calc_angle(v1, v2, v3)
        theta_list.append(theta)
        res = ca_list[i + 1].get_parent()
        res.xtra['THETA'] = theta
    return theta_list