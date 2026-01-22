from Bio.Data import IUPACData
from typing import Dict, List, Optional
def register_ncbi_table(name, alt_name, id, table, start_codons, stop_codons):
    """Turn codon table data into objects (PRIVATE).

    The data is stored in the dictionaries.
    """
    names = [x.strip() for x in name.replace(' and ', '; ').replace(', ', '; ').split('; ')]
    dna = NCBICodonTableDNA(id, names + [alt_name], table, start_codons, stop_codons)
    ambig_dna = AmbiguousCodonTable(dna, IUPACData.ambiguous_dna_letters, IUPACData.ambiguous_dna_values, IUPACData.extended_protein_letters, IUPACData.extended_protein_values)
    rna_table = {}
    generic_table = {}
    for codon, val in table.items():
        generic_table[codon] = val
        codon = codon.replace('T', 'U')
        generic_table[codon] = val
        rna_table[codon] = val
    rna_start_codons = []
    generic_start_codons = []
    for codon in start_codons:
        generic_start_codons.append(codon)
        if 'T' in codon:
            codon = codon.replace('T', 'U')
            generic_start_codons.append(codon)
        rna_start_codons.append(codon)
    rna_stop_codons = []
    generic_stop_codons = []
    for codon in stop_codons:
        generic_stop_codons.append(codon)
        if 'T' in codon:
            codon = codon.replace('T', 'U')
            generic_stop_codons.append(codon)
        rna_stop_codons.append(codon)
    generic = NCBICodonTable(id, names + [alt_name], generic_table, generic_start_codons, generic_stop_codons)
    _merged_values = dict(IUPACData.ambiguous_rna_values.items())
    _merged_values['T'] = 'U'
    ambig_generic = AmbiguousCodonTable(generic, None, _merged_values, IUPACData.extended_protein_letters, IUPACData.extended_protein_values)
    rna = NCBICodonTableRNA(id, names + [alt_name], rna_table, rna_start_codons, rna_stop_codons)
    ambig_rna = AmbiguousCodonTable(rna, IUPACData.ambiguous_rna_letters, IUPACData.ambiguous_rna_values, IUPACData.extended_protein_letters, IUPACData.extended_protein_values)
    if id == 1:
        global standard_dna_table, standard_rna_table
        standard_dna_table = dna
        standard_rna_table = rna
    unambiguous_dna_by_id[id] = dna
    unambiguous_rna_by_id[id] = rna
    generic_by_id[id] = generic
    ambiguous_dna_by_id[id] = ambig_dna
    ambiguous_rna_by_id[id] = ambig_rna
    ambiguous_generic_by_id[id] = ambig_generic
    if alt_name is not None:
        names.append(alt_name)
    for name in names:
        unambiguous_dna_by_name[name] = dna
        unambiguous_rna_by_name[name] = rna
        generic_by_name[name] = generic
        ambiguous_dna_by_name[name] = ambig_dna
        ambiguous_rna_by_name[name] = ambig_rna
        ambiguous_generic_by_name[name] = ambig_generic