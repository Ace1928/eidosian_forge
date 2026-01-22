from plotly.basedatatypes import BaseFigure
def for_each_selection(self, fn, selector=None, row=None, col=None, secondary_y=None):
    """
        Apply a function to all selections that satisfy the specified selection
        criteria

        Parameters
        ----------
        fn:
            Function that inputs a single selection object.
        selector: dict, function, int, str or None (default None)
            Dict to use as selection criteria.
            Traces will be selected if they contain properties corresponding
            to all of the dictionary's keys, with values that exactly match
            the supplied values. If None (the default), all selections are
            selected. If a function, it must be a function accepting a single
            argument and returning a boolean. The function will be called on
            each selection and those for which the function returned True
            will be in the selection. If an int N, the Nth selection matching row
            and col will be selected (N can be negative). If a string S, the selector
            is equivalent to dict(type=S).
        row, col: int or None (default None)
            Subplot row and column index of selections to select.
            To select selections by row and column, the Figure must have been
            created using plotly.subplots.make_subplots.  To select only those
            selections that are in paper coordinates, set row and col to the
            string 'paper'.  If None (the default), all selections are selected.
        secondary_y: boolean or None (default None)
            * If True, only select selections associated with the secondary
              y-axis of the subplot.
            * If False, only select selections associated with the primary
              y-axis of the subplot.
            * If None (the default), do not filter selections based on secondary
              y-axis.

            To select selections by secondary y-axis, the Figure must have been
            created using plotly.subplots.make_subplots. See the docstring
            for the specs argument to make_subplots for more info on
            creating subplots with secondary y-axes.
        Returns
        -------
        self
            Returns the Figure object that the method was called on
        """
    for obj in self._select_annotations_like(prop='selections', selector=selector, row=row, col=col, secondary_y=secondary_y):
        fn(obj)
    return self