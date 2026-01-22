from Bio.PopGen.GenePop import get_indiv
def get_individual(self):
    """Get the next individual.

        Returns individual information if there are more individuals
        in the current population.
        Returns True if there are no more individuals in the current
        population, but there are more populations. Next read will
        be of the following pop.
        Returns False if at end of file.
        """
    for line in self._handle:
        line = line.rstrip()
        if line.upper() == 'POP':
            self.current_pop += 1
            self.current_ind = 0
            return True
        else:
            self.current_ind += 1
            indiv_name, allele_list, ignore = get_indiv(line)
            return (indiv_name, allele_list)
    return False