@property
def short_id(self):
    """
        The ID of the object, truncated to 12 characters.
        """
    return self.id[:12]