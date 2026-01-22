from plotly.utils import _list_repr_elided
@property
def point_inds(self):
    """
        List of selected indexes into the trace's points

        Returns
        -------
        list[int]
        """
    return self._point_inds