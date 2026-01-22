from twisted.python import reflect
def listStaticEntities(self):
    """Retrieve a list of all name, entity pairs that I store references to.

        See getStaticEntity.
        """
    return self.entities.items()