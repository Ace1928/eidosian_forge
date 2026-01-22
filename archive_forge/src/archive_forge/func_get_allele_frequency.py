from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_allele_frequency(self, pop_pos, locus_name):
    """Calculate the allele frequency for a certain locus on a population."""
    if len(self.__allele_frequency) == 0:
        geno_freqs = self._controller.calc_allele_genotype_freqs(self._fname)
        pop_iter, loc_iter = geno_freqs
        for locus_info in loc_iter:
            if locus_info[0] is None:
                self.__allele_frequency[locus_info[0]] = (None, None)
            else:
                self.__allele_frequency[locus_info[0]] = locus_info[1:]
    info = self.__allele_frequency[locus_name]
    pop_name, freqs, total = info[1][pop_pos]
    allele_freq = {}
    alleles = info[0]
    for i, allele in enumerate(alleles):
        allele_freq[allele] = freqs[i]
    return (total, allele_freq)