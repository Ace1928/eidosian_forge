from Bio.PopGen.GenePop import get_indiv
def remove_loci_by_name(self, names, fname):
    """Remove a loci list (by name).

        Arguments:
         - names - names
         - fname - file to be created with loci removed

        """
    positions = []
    for i, locus in enumerate(self.loci_list):
        if locus in names:
            positions.append(i)
    self.remove_loci_by_position(positions, fname)