from copy import deepcopy
def split_in_pops(self, pop_names):
    """Split a GP record in a dictionary with 1 pop per entry.

        Given a record with n pops and m loci returns a dictionary
        of records (key pop_name) where each item is a record
        with a single pop and m loci.

        Arguments:
        - pop_names - Population names

        """
    gp_pops = {}
    for i, population in enumerate(self.populations):
        gp_pop = Record()
        gp_pop.marker_len = self.marker_len
        gp_pop.comment_line = self.comment_line
        gp_pop.loci_list = deepcopy(self.loci_list)
        gp_pop.populations = [deepcopy(population)]
        gp_pops[pop_names[i]] = gp_pop
    return gp_pops