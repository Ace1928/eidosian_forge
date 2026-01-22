from twisted.python import reflect
def delEntity(self, name):
    """Remove a static reference for 'name'.

        Raises a KeyError if the operation fails.
        """
    del self.entities[name]