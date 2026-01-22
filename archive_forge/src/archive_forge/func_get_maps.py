from Bio.Data import PDBData
from Bio.PDB import Selection
from Bio.PDB.Polypeptide import is_aa
def get_maps(self):
    """Map residues between the structures.

        Return two dictionaries that map a residue in one structure to
        the equivealent residue in the other structure.
        """
    return (self.map12, self.map21)