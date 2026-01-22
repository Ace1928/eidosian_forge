from math import sqrt, erfc
import warnings
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio import BiopythonWarning
from Bio.codonalign.codonseq import _get_codon_list, CodonSeq, cal_dn_ds
def get_dn_ds_tree(self, dn_ds_method='NG86', tree_method='UPGMA', codon_table=None):
    """Construct dn tree and ds tree.

        Argument:
         - dn_ds_method - Available methods include NG86, LWL85, YN00 and ML.
         - tree_method  - Available methods include UPGMA and NJ.

        """
    from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
    if codon_table is None:
        codon_table = CodonTable.generic_by_id[1]
    dn_dm, ds_dm = self.get_dn_ds_matrix(method=dn_ds_method, codon_table=codon_table)
    dn_constructor = DistanceTreeConstructor()
    ds_constructor = DistanceTreeConstructor()
    if tree_method == 'UPGMA':
        dn_tree = dn_constructor.upgma(dn_dm)
        ds_tree = ds_constructor.upgma(ds_dm)
    elif tree_method == 'NJ':
        dn_tree = dn_constructor.nj(dn_dm)
        ds_tree = ds_constructor.nj(ds_dm)
    else:
        raise RuntimeError(f'Unknown tree method ({tree_method}). Only NJ and UPGMA are accepted.')
    return (dn_tree, ds_tree)