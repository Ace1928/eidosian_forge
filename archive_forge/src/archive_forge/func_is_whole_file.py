def is_whole_file(self):
    """Returns True if this range includes all possible bytes.

        This can only occur if the 'last' member is None and the first
        member is 0.

        """
    return self.first == 0 and self.last is None