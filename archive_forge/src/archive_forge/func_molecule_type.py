import re
import warnings
from Bio import BiopythonParserWarning
from Bio.Seq import Seq
from Bio.SeqFeature import Location
from Bio.SeqFeature import Reference
from Bio.SeqFeature import SeqFeature
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import LocationParserError
from .utils import FeatureValueCleaner
from .Scanner import GenBankScanner
def molecule_type(self, mol_type):
    """Validate and record the molecule type (for round-trip etc)."""
    if mol_type:
        if 'circular' in mol_type or 'linear' in mol_type:
            raise ParserFailureError(f'Molecule type {mol_type!r} should not include topology')
        if mol_type[-3:].upper() in ('DNA', 'RNA') and (not mol_type[-3:].isupper()):
            warnings.warn(f'Non-upper case molecule type in LOCUS line: {mol_type}', BiopythonParserWarning)
        self.data.molecule_type = mol_type