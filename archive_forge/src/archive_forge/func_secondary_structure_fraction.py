import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def secondary_structure_fraction(self):
    """Calculate fraction of helix, turn and sheet.

        Returns a list of the fraction of amino acids which tend
        to be in Helix, Turn or Sheet, according to Haimov and Srebnik, 2016;
        Hutchinson and Thornton, 1994; and Kim and Berg, 1993, respectively.

        Amino acids in helix: E, M, A, L, K.
        Amino acids in turn: N, P, G, S, D.
        Amino acids in sheet: V, I, Y, F, W, L, T.

        Note that, prior to v1.82, this method wrongly returned
        (Sheet, Turn, Helix) while claiming to return (Helix, Turn, Sheet).

        Returns a tuple of three floats (Helix, Turn, Sheet).
        """
    aa_percentages = self.get_amino_acids_percent()
    helix = sum((aa_percentages[r] for r in 'EMALK'))
    turn = sum((aa_percentages[r] for r in 'NPGSD'))
    sheet = sum((aa_percentages[r] for r in 'VIYFWLT'))
    return (helix, turn, sheet)