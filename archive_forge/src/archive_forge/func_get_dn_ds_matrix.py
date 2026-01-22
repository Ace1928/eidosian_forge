from math import sqrt, erfc
import warnings
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio import BiopythonWarning
from Bio.codonalign.codonseq import _get_codon_list, CodonSeq, cal_dn_ds
def get_dn_ds_matrix(self, method='NG86', codon_table=None):
    """Available methods include NG86, LWL85, YN00 and ML.

        Argument:
         - method       - Available methods include NG86, LWL85, YN00 and ML.
         - codon_table  - Codon table to use for forward translation.

        """
    from Bio.Phylo.TreeConstruction import DistanceMatrix as DM
    if codon_table is None:
        codon_table = CodonTable.generic_by_id[1]
    names = [i.id for i in self._records]
    size = len(self._records)
    dn_matrix = []
    ds_matrix = []
    for i in range(size):
        dn_matrix.append([])
        ds_matrix.append([])
        for j in range(i + 1):
            if i != j:
                dn, ds = cal_dn_ds(self._records[i], self._records[j], method=method, codon_table=codon_table)
                dn_matrix[i].append(dn)
                ds_matrix[i].append(ds)
            else:
                dn_matrix[i].append(0.0)
                ds_matrix[i].append(0.0)
    dn_dm = DM(names, matrix=dn_matrix)
    ds_dm = DM(names, matrix=ds_matrix)
    return (dn_dm, ds_dm)