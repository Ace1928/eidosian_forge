from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_heterozygosity_info(self, pop_pos, locus_name):
    """Return the heterozygosity info for a certain locus on a population.

        Returns (Expected homozygotes, observed homozygotes,
        Expected heterozygotes, observed heterozygotes)
        """
    geno_freqs = self._controller.calc_allele_genotype_freqs(self._fname)
    pop_iter, loc_iter = geno_freqs
    pops = list(pop_iter)
    return pops[pop_pos][1][locus_name][1]