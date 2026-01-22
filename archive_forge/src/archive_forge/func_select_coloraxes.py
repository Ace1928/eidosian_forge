from plotly.basedatatypes import BaseFigure
def select_coloraxes(self, selector=None, row=None, col=None):
    """
        Select coloraxis subplot objects from a particular subplot cell
        and/or coloraxis subplot objects that satisfy custom selection
        criteria.

        Parameters
        ----------
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            coloraxis objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all coloraxis objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            coloraxis and those for which the function returned True will
            be in the selection.
        row, col: int or None (default None)
            Subplot row and column index of coloraxis objects to select.
            To select coloraxis objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all coloraxis objects are selected.
        Returns
        -------
        generator
            Generator that iterates through all of the coloraxis
            objects that satisfy all of the specified selection criteria
        """
    return self._select_layout_subplots_by_prefix('coloraxis', selector, row, col)