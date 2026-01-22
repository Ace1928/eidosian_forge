from plotly.basedatatypes import BaseFigure
def select_scenes(self, selector=None, row=None, col=None):
    """
        Select scene subplot objects from a particular subplot cell
        and/or scene subplot objects that satisfy custom selection
        criteria.

        Parameters
        ----------
        selector: dict, function, or None (default None)
            Dict to use as selection criteria.
            scene objects will be selected if they contain
            properties corresponding to all of the dictionary's keys, with
            values that exactly match the supplied values. If None
            (the default), all scene objects are selected. If a
            function, it must be a function accepting a single argument and
            returning a boolean. The function will be called on each
            scene and those for which the function returned True will
            be in the selection.
        row, col: int or None (default None)
            Subplot row and column index of scene objects to select.
            To select scene objects by row and column, the Figure
            must have been created using plotly.subplots.make_subplots.
            If None (the default), all scene objects are selected.
        Returns
        -------
        generator
            Generator that iterates through all of the scene
            objects that satisfy all of the specified selection criteria
        """
    return self._select_layout_subplots_by_prefix('scene', selector, row, col)