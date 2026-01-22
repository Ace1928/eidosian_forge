@property
def combinable(self):
    """
        Whether this type of value may be combined with other values
        for the same field. This makes the filter potentially more efficient
        """
    return False