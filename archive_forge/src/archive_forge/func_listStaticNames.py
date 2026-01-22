from twisted.python import reflect
def listStaticNames(self):
    """Retrieve a list of the names of entities that I store references to.

        See getStaticEntity.
        """
    return self.entities.keys()