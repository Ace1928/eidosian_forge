Advance the global stale flag, marking all variables as stale

        This is generally called immediately before and after a batch
        variable update (i.e. loading values from a solver result or
        stored solution).  Before the batch update
        :meth:`mark_all_as_stale` is called with ``delayed=False``,
        which immediately marks all variables as stale.  After the batch
        update, :meth:`mark_all_as_stale` is typically called with
        ``delayed=True``.  This allows additional stale variables to be
        updated without advancing the global flag, but as soon as any
        non-stale variable has its value changed, then the flag is
        advanced and all other variables become stale.

        