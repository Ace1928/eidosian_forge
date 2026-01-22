import itertools
from collections import defaultdict
from string import ascii_uppercase
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import Select, StructureIO
from mmtf.api.mmtf_writer import MMTFEncoder
from Bio.SeqUtils import seq1
from Bio.Data.PDBData import protein_letters_3to1_extended
Label chains sequentially: A, B, ..., Z, AA, AB etc.