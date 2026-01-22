from plotly.basedatatypes import BaseFigure
def for_each_xaxis(self, fn, selector=None, row=None, col=None) -> 'Figure':
    """
        Apply a function to all xaxis objects that satisfy the
        specified selection criteria

        Parameters
        ----------
        fn:
            Function that inputs a single xaxis object.
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            xaxis objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all xaxis objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            xaxis and those for which the function returned True will
            be in the selection.
        row, col: int or None (default None)
            Subplot row and column index of xaxis objects to select.
            To select xaxis objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all xaxis objects are selected.
        Returns
        -------
        self
            Returns the Figure object that the method was called on
        """
    for obj in self.select_xaxes(selector=selector, row=row, col=col):
        fn(obj)
    return self