@property
def found_violation(self):
    """
        bool : True if ``found_violation`` attribute for
        main separation loop results is True, False otherwise.
        """
    found_viol = self.get_violating_attr('found_violation')
    if found_viol is None:
        found_viol = False
    return found_viol