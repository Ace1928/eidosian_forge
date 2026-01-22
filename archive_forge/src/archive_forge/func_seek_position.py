from Bio.PopGen.GenePop import get_indiv
def seek_position(self, pop, indiv):
    """Seek a certain position in the file.

        Arguments:
         - pop - pop position (0 is first)
         - indiv - individual in pop

        """
    self._handle.seek(0)
    self.skip_header()
    while pop > 0:
        self.skip_population()
        pop -= 1
    while indiv > 0:
        self.get_individual()
        indiv -= 1