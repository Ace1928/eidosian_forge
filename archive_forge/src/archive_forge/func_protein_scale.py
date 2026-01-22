import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def protein_scale(self, param_dict, window, edge=1.0):
    """Compute a profile by any amino acid scale.

        An amino acid scale is defined by a numerical value assigned to each
        type of amino acid. The most frequently used scales are the
        hydrophobicity or hydrophilicity scales and the secondary structure
        conformational parameters scales, but many other scales exist which
        are based on different chemical and physical properties of the
        amino acids.  You can set several parameters that control the
        computation of a scale profile, such as the window size and the window
        edge relative weight value.

        WindowSize: The window size is the length of the interval to use for
        the profile computation. For a window size n, we use the i-(n-1)/2
        neighboring residues on each side to compute the score for residue i.
        The score for residue i is the sum of the scaled values for these
        amino acids, optionally weighted according to their position in the
        window.

        Edge: The central amino acid of the window always has a weight of 1.
        By default, the amino acids at the remaining window positions have the
        same weight, but you can make the residue at the center of the window
        have a larger weight than the others by setting the edge value for the
        residues at the beginning and end of the interval to a value between
        0 and 1. For instance, for Edge=0.4 and a window size of 5 the weights
        will be: 0.4, 0.7, 1.0, 0.7, 0.4.

        The method returns a list of values which can be plotted to view the
        change along a protein sequence.  Many scales exist. Just add your
        favorites to the ProtParamData modules.

        Similar to expasy's ProtScale:
        http://www.expasy.org/cgi-bin/protscale.pl
        """
    weights = self._weight_list(window, edge)
    scores = []
    sum_of_weights = sum(weights) * 2 + 1
    for i in range(self.length - window + 1):
        subsequence = self.sequence[i:i + window]
        score = 0.0
        for j in range(window // 2):
            try:
                front = param_dict[subsequence[j]]
                back = param_dict[subsequence[window - j - 1]]
                score += weights[j] * front + weights[j] * back
            except KeyError:
                sys.stderr.write('warning: %s or %s is not a standard amino acid.\n' % (subsequence[j], subsequence[window - j - 1]))
        middle = subsequence[window // 2]
        if middle in param_dict:
            score += param_dict[middle]
        else:
            sys.stderr.write(f'warning: {middle} is not a standard amino acid.\n')
        scores.append(score / sum_of_weights)
    return scores