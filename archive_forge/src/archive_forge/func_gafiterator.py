import copy
def gafiterator(handle):
    """Iterate over a GAF 1.0 or 2.x file.

    This function should be called to read a
    gene_association.goa_uniprot file. Reads the first record and
    returns a gaf 2.x or a gaf 1.0 iterator as needed

    Example: open, read, interat and filter results.

    Original data file has been trimmed to ~600 rows.

    Original source ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/YEAST/goa_yeast.gaf.gz

    >>> from Bio.UniProt.GOA import gafiterator, record_has
    >>> Evidence = {'Evidence': set(['ND'])}
    >>> Synonym = {'Synonym': set(['YA19A_YEAST', 'YAL019W-A'])}
    >>> Taxon_ID = {'Taxon_ID': set(['taxon:559292'])}
    >>> with open('UniProt/goa_yeast.gaf', 'r') as handle:
    ...     for rec in gafiterator(handle):
    ...         if record_has(rec, Taxon_ID) and record_has(rec, Evidence) and record_has(rec, Synonym):
    ...             for key in ('DB_Object_Name', 'Evidence', 'Synonym', 'Taxon_ID'):
    ...                 print(rec[key])
    ...
    Putative uncharacterized protein YAL019W-A
    ND
    ['YA19A_YEAST', 'YAL019W-A']
    ['taxon:559292']
    Putative uncharacterized protein YAL019W-A
    ND
    ['YA19A_YEAST', 'YAL019W-A']
    ['taxon:559292']
    Putative uncharacterized protein YAL019W-A
    ND
    ['YA19A_YEAST', 'YAL019W-A']
    ['taxon:559292']

    """
    inline = handle.readline()
    if inline.strip() == '!gaf-version: 2.0':
        return _gaf20iterator(handle)
    elif inline.strip() == '!gaf-version: 2.1':
        return _gaf20iterator(handle)
    elif inline.strip() == '!gaf-version: 2.2':
        return _gaf20iterator(handle)
    elif inline.strip() == '!gaf-version: 1.0':
        return _gaf10iterator(handle)
    else:
        raise ValueError(f'Unknown GAF version {inline}\n')