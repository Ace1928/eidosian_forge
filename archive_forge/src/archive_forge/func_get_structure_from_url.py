from Bio.PDB.mmtf.DefaultParser import StructureDecoder
from .mmtfio import MMTFIO
@staticmethod
def get_structure_from_url(pdb_id):
    """Get a structure from a URL - given a PDB id.

        :param pdb_id: the input PDB id
        :return: the structure

        """
    decoder = fetch(pdb_id)
    return get_from_decoded(decoder)