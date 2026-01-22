def get_run_iterator(self):
    """Get an extended iterator over the run list.

        :rtype: `RunIterator`
        """
    return RunIterator(self)