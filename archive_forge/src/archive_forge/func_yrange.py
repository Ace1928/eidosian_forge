from plotly.utils import _list_repr_elided
@property
def yrange(self):
    """
        y-axis range extents of the box selection

        Returns
        -------
        (float, float)
        """
    return self._yrange