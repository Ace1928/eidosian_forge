import itertools
def check_subset(self, subset_object):
    """
        Check if subset_object is a subset of the current measurement object

        Parameters
        ----------
        subset_object: a measurement object
        """
    for name in subset_object.variable_names:
        if name not in self.variable_names:
            raise ValueError('Measurement not in the set: ', name)
    return True