from Bio.PDB.mmtf.DefaultParser import StructureDecoder
from .mmtfio import MMTFIO
def get_from_decoded(decoder):
    """Return structure from decoder."""
    structure_decoder = StructureDecoder()
    decoder.pass_data_on(structure_decoder)
    return structure_decoder.structure_builder.get_structure()