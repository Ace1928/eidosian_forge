import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def flexibility(self):
    """Calculate the flexibility according to Vihinen, 1994.

        No argument to change window size because parameters are specific for
        a window=9. The parameters used are optimized for determining the
        flexibility.
        """
    flexibilities = ProtParamData.Flex
    window_size = 9
    weights = [0.25, 0.4375, 0.625, 0.8125, 1]
    scores = []
    for i in range(self.length - window_size):
        subsequence = self.sequence[i:i + window_size]
        score = 0.0
        for j in range(window_size // 2):
            front = subsequence[j]
            back = subsequence[window_size - j - 1]
            score += (flexibilities[front] + flexibilities[back]) * weights[j]
        middle = subsequence[window_size // 2 + 1]
        score += flexibilities[middle]
        scores.append(score / 5.25)
    return scores