from plotly.basedatatypes import BaseFigure
def for_each_trace(self, fn, selector=None, row=None, col=None, secondary_y=None) -> 'Figure':
    """

        Apply a function to all traces that satisfy the specified selection
        criteria

        Parameters
        ----------
        fn:
            Function that inputs a single trace object.
        selector: dict, function, int, str or None (default None)
            Dict to use as selection criteria.
            Traces will be selected if they contain properties corresponding
            to all of the dictionary's keys, with values that exactly match
            the supplied values. If None (the default), all traces are
            selected. If a function, it must be a function accepting a single
            argument and returning a boolean. The function will be called on
            each trace and those for which the function returned True
            will be in the selection. If an int N, the Nth trace matching row
            and col will be selected (N can be negative). If a string S, the selector
            is equivalent to dict(type=S).
        row, col: int or None (default None)
            Subplot row and column index of traces to select.
            To select traces by row and column, the Figure must have been
            created using plotly.subplots.make_subplots.  If None
            (the default), all traces are selected.
        secondary_y: boolean or None (default None)
            * If True, only select traces associated with the secondary
              y-axis of the subplot.
            * If False, only select traces associated with the primary
              y-axis of the subplot.
            * If None (the default), do not filter traces based on secondary
              y-axis.

            To select traces by secondary y-axis, the Figure must have been
            created using plotly.subplots.make_subplots. See the docstring
            for the specs argument to make_subplots for more info on
            creating subplots with secondary y-axes.
        Returns
        -------
        self
            Returns the Figure object that the method was called on

        """
    return super(Figure, self).for_each_trace(fn, selector, row, col, secondary_y)