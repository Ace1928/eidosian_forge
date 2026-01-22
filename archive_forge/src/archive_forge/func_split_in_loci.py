from copy import deepcopy
def split_in_loci(self, gp):
    """Split a GP record in a dictionary with 1 locus per entry.

        Given a record with n pops and m loci returns a dictionary
        of records (key locus name) where each item is a record
        with a single locus and n pops.
        """
    gp_loci = {}
    for i, locus in enumerate(self.loci_list):
        gp_pop = Record()
        gp_pop.marker_len = self.marker_len
        gp_pop.comment_line = self.comment_line
        gp_pop.loci_list = [locus]
        gp_pop.populations = []
        for pop in self.populations:
            my_pop = []
            for indiv in pop:
                my_pop.append((indiv[0], [indiv[1][i]]))
            gp_pop.populations.append(my_pop)
        gp_loci[gp_pop.loci_list[0]] = gp_pop
    return gp_loci