import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def molecular_weight(self):
    """Calculate MW from Protein sequence."""
    return molecular_weight(self.sequence, seq_type='protein', monoisotopic=self.monoisotopic)