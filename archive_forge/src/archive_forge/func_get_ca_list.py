import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def get_ca_list(self):
    """Get list of C-alpha atoms in the polypeptide.

        :return: the list of C-alpha atoms
        :rtype: [L{Atom}, L{Atom}, ...]
        """
    ca_list = []
    for res in self:
        ca = res['CA']
        ca_list.append(ca)
    return ca_list