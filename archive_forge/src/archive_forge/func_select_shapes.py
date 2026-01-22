from plotly.basedatatypes import BaseFigure
def select_shapes(self, selector=None, row=None, col=None, secondary_y=None):
    """
        Select shapes from a particular subplot cell and/or shapes
        that satisfy custom selection criteria.

        Parameters
        ----------
        selector: dict, function, int, str, or None (default None)
            Dict to use as selection criteria.
            Annotations will be selected if they contain properties corresponding
            to all of the dictionary's keys, with values that exactly match
            the supplied values. If None (the default), all shapes are
            selected. If a function, it must be a function accepting a single
            argument and returning a boolean. The function will be called on
            each shape and those for which the function returned True
            will be in the selection. If an int N, the Nth shape matching row
            and col will be selected (N can be negative). If a string S, the selector
            is equivalent to dict(type=S).
        row, col: int or None (default None)
            Subplot row and column index of shapes to select.
            To select shapes by row and column, the Figure must have been
            created using plotly.subplots.make_subplots.  To select only those
            shape that are in paper coordinates, set row and col to the
            string 'paper'.  If None (the default), all shapes are selected.
        secondary_y: boolean or None (default None)
            * If True, only select shapes associated with the secondary
              y-axis of the subplot.
            * If False, only select shapes associated with the primary
              y-axis of the subplot.
            * If None (the default), do not filter shapes based on secondary
              y-axis.

            To select shapes by secondary y-axis, the Figure must have been
            created using plotly.subplots.make_subplots. See the docstring
            for the specs argument to make_subplots for more info on
            creating subplots with secondary y-axes.
        Returns
        -------
        generator
            Generator that iterates through all of the shapes that satisfy
            all of the specified selection criteria
        """
    return self._select_annotations_like('shapes', selector=selector, row=row, col=col, secondary_y=secondary_y)