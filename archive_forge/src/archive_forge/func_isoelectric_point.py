import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def isoelectric_point(self):
    """Calculate the isoelectric point.

        Uses the module IsoelectricPoint to calculate the pI of a protein.
        """
    aa_content = self.count_amino_acids()
    ie_point = IsoelectricPoint.IsoelectricPoint(self.sequence, aa_content)
    return ie_point.pi()