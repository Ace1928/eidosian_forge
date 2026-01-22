import io
from urllib.request import urlopen
import time
from Bio._utils import function_with_previous
def kegg_conv(target_db, source_db, option=None):
    """KEGG conv - convert KEGG identifiers to/from outside identifiers.

    Arguments:
     - target_db - Target database
     - source_db_or_dbentries - source database or database entries
     - option - Can be "turtle" or "n-triple" (string).

    """
    if option and option not in ['turtle', 'n-triple']:
        raise ValueError('Invalid option arg for kegg conv request.')
    if isinstance(source_db, list):
        source_db = '+'.join(source_db)
    if target_db in ['ncbi-gi', 'ncbi-geneid', 'uniprot'] or source_db in ['ncbi-gi', 'ncbi-geneid', 'uniprot'] or (target_db in ['drug', 'compound', 'glycan'] and source_db in ['pubchem', 'glycan']) or (target_db in ['pubchem', 'glycan'] and source_db in ['drug', 'compound', 'glycan']):
        if option:
            resp = _q('conv', target_db, source_db, option)
        else:
            resp = _q('conv', target_db, source_db)
        return resp
    else:
        raise ValueError('Bad argument target_db or source_db for kegg conv request.')