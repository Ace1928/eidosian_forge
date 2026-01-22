@property
def until_line(self):
    """The line where the error ends (starting with 1)."""
    return self._parso_error.end_pos[0]