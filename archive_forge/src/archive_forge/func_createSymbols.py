from weakref import ref as weakref_ref
def createSymbols(self, objs, labeler=None, *args):
    """
        Create a symbol for iterable objects with a given labeler.  No
        error checking is done to ensure that the generated symbol
        names are unique.
        """
    if labeler is None:
        if self.default_labeler is not None:
            labeler = self.default_labeler
        else:
            labeler = str
    self.addSymbols(((obj, labeler(obj, *args)) for obj in objs))