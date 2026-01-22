import enum
def visitDict(self, obj, *args, **kwargs):
    """Called to visit any value that is a dictionary."""
    for value in obj.values():
        self.visit(value, *args, **kwargs)