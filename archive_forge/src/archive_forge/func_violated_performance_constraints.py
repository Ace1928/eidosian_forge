@property
def violated_performance_constraints(self):
    """
        Return list of violated performance constraints.
        """
    return self.get_violating_attr('violated_performance_constraints')