import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def get_amino_acids_percent(self):
    """Calculate the amino acid content in percentages.

        The same as count_amino_acids only returns the Number in percentage of
        entire sequence. Returns a dictionary of {AminoAcid:percentage}.

        The return value is cached in self.amino_acids_percent.

        input is the dictionary self.amino_acids_content.
        output is a dictionary with amino acids as keys.
        """
    if self.amino_acids_percent is None:
        aa_counts = self.count_amino_acids()
        percentages = {aa: count / self.length for aa, count in aa_counts.items()}
        self.amino_acids_percent = percentages
    return self.amino_acids_percent