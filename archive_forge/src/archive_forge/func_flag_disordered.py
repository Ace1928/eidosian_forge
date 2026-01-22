from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Entity import Entity, DisorderedEntityWrapper
def flag_disordered(self):
    """Set the disordered flag."""
    self.disordered = 1