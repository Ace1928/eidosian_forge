from plotly.basedatatypes import BaseFigure
def select_yaxes(self, selector=None, row=None, col=None, secondary_y=None):
    """
        Select yaxis subplot objects from a particular subplot cell
        and/or yaxis subplot objects that satisfy custom selection
        criteria.

        Parameters
        ----------
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            yaxis objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all yaxis objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            yaxis and those for which the function returned True will
            be in the selection.
        row, col: int or None (default None)
            Subplot row and column index of yaxis objects to select.
            To select yaxis objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all yaxis objects are selected.
        secondary_y: boolean or None (default None)
            * If True, only select yaxis objects associated with the secondary
              y-axis of the subplot.
            * If False, only select yaxis objects associated with the primary
              y-axis of the subplot.
            * If None (the default), do not filter yaxis objects based on
              a secondary y-axis condition.

            To select yaxis objects by secondary y-axis, the Figure must
            have been created using plotly.subplots.make_subplots. See
            the docstring for the specs argument to make_subplots for more
            info on creating subplots with secondary y-axes.
        Returns
        -------
        generator
            Generator that iterates through all of the yaxis
            objects that satisfy all of the specified selection criteria
        """
    return self._select_layout_subplots_by_prefix('yaxis', selector, row, col, secondary_y=secondary_y)