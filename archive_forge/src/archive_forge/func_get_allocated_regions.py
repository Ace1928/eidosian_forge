def get_allocated_regions(self):
    """Get a list of (aggregate) allocated regions.

        The result of this method is ``(starts, sizes)``, where ``starts`` is
        a list of starting indices of the regions and ``sizes`` their
        corresponding lengths.

        :rtype: (list, list)
        """
    return (self.starts, self.sizes)