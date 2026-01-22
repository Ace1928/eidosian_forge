from twisted.python import reflect
def nameConstraint(self, name):
    """A method that determines whether an entity may be added to me with a given name.

        If the constraint is satisfied, return 1; if the constraint is not
        satisfied, either return 0 or raise a descriptive ConstraintViolation.
        """
    return 1