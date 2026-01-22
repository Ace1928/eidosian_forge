def get_free_size(self):
    """Return the amount of space unused.
        
        :rtype: int
        """
    if not self.starts:
        return self.capacity
    free_end = self.capacity - (self.starts[-1] + self.sizes[-1])
    return self.get_fragmented_free_size() + free_end