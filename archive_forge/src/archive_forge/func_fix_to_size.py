def fix_to_size(self, size):
    """Changes all length-relative range_specs to absolute range_specs based upon given file size.
        If none of the range_specs in this set can be satisfied, then the
        entire set is considered unsatifiable and an error is raised.
        Otherwise any unsatisfiable range_specs will simply be removed
        from this set.

        """
    for i in range(len(self.range_specs)):
        try:
            self.range_specs[i].fix_to_size(size)
        except RangeUnsatisfiableError:
            self.range_specs[i] = None
    self.range_specs = [s for s in self.range_specs if s is not None]
    if len(self.range_specs) == 0:
        raise RangeUnsatisfiableError('No ranges can be satisfied')