import itertools
from Bio.PDB.Atom import Atom
from Bio.PDB.Entity import Entity
from Bio.PDB.PDBExceptions import PDBException
def uniqueify(items):
    """Return a list of the unique items in the given iterable.

    Order is NOT preserved.
    """
    return list(set(items))