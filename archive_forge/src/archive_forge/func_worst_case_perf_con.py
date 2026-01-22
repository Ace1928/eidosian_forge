@property
def worst_case_perf_con(self):
    """
        ConstraintData : Performance constraint corresponding to the
        separation solution chosen for the next master problem.
        """
    return self.get_violating_attr('worst_case_perf_con')