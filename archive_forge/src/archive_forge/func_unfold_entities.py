import itertools
from Bio.PDB.Atom import Atom
from Bio.PDB.Entity import Entity
from Bio.PDB.PDBExceptions import PDBException
def unfold_entities(entity_list, target_level):
    """Unfold entities list to a child level (e.g. residues in chain).

    Unfold a list of entities to a list of entities of another
    level.  E.g.:

    list of atoms -> list of residues
    list of modules -> list of atoms
    list of residues -> list of chains

    - entity_list - list of entities or a single entity
    - target_level - char (A, R, C, M, S)

    Note that if entity_list is an empty list, you get an empty list back:

    >>> unfold_entities([], "A")
    []

    """
    if target_level not in entity_levels:
        raise PDBException(f'{target_level}: Not an entity level.')
    if entity_list == []:
        return []
    if isinstance(entity_list, (Entity, Atom)):
        entity_list = [entity_list]
    level = entity_list[0].get_level()
    if not all((entity.get_level() == level for entity in entity_list)):
        raise PDBException('Entity list is not homogeneous.')
    target_index = entity_levels.index(target_level)
    level_index = entity_levels.index(level)
    if level_index == target_index:
        return entity_list
    entities = entity_list
    if level_index > target_index:
        for i in range(target_index, level_index):
            entities = itertools.chain.from_iterable(entities)
    else:
        for i in range(level_index, target_index):
            entities = {entity.get_parent(): None for entity in entities}
    return list(entities)