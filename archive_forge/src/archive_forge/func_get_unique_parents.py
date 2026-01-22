import itertools
from Bio.PDB.Atom import Atom
from Bio.PDB.Entity import Entity
from Bio.PDB.PDBExceptions import PDBException
def get_unique_parents(entity_list):
    """Translate a list of entities to a list of their (unique) parents."""
    unique_parents = {entity.get_parent() for entity in entity_list}
    return list(unique_parents)