from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_genotype_count(self, pop_pos, locus_name):
    """Return the genotype counts for a certain population and locus."""
    geno_freqs = self._controller.calc_allele_genotype_freqs(self._fname)
    pop_iter, loc_iter = geno_freqs
    pop_iter = list(pop_iter)
    return pop_iter[pop_pos][1][locus_name][0]