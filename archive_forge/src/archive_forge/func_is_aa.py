import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def is_aa(residue, standard=False):
    """Return True if residue object/string is an amino acid.

    :param residue: a L{Residue} object OR a three letter amino acid code
    :type residue: L{Residue} or string

    :param standard: flag to check for the 20 AA (default false)
    :type standard: boolean

    >>> is_aa('ALA')
    True

    Known three letter codes for modified amino acids are supported,

    >>> is_aa('FME')
    True
    >>> is_aa('FME', standard=True)
    False
    """
    if not isinstance(residue, str):
        residue = f'{residue.get_resname():<3s}'
    residue = residue.upper()
    if standard:
        return residue in protein_letters_3to1
    else:
        return residue in protein_letters_3to1_extended