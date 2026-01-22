def getMapping(self):
    """Returns the mapping from things to their class set.

        The return value belongs to the Classifier object and should NOT
        be modified while the classifier is still in use.
        """
    self._process()
    return self._mapping