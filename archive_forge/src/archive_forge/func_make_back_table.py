from Bio.Data import IUPACData
from typing import Dict, List, Optional
def make_back_table(table, default_stop_codon):
    """Back a back-table (naive single codon mapping).

    ONLY RETURNS A SINGLE CODON, chosen from the possible alternatives
    based on their sort order.
    """
    back_table = {}
    for key in sorted(table):
        back_table[table[key]] = key
    back_table[None] = default_stop_codon
    return back_table