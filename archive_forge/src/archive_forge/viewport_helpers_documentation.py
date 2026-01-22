import math
from ..bindings.view_state import ViewState
from .type_checking import is_pandas_df
Automatically computes a zoom level for the points passed in.

    Parameters
    ----------
    points : list of list of float or pandas.DataFrame
        A list of points
    view_propotion : float, default 1
        Proportion of the data that is meaningful to plot
    view_type : class constructor for pydeck.ViewState, default :class:`pydeck.bindings.view_state.ViewState`
        Class constructor for a viewport. In the current version of pydeck,
        users most likely do not have to modify this attribute.

    Returns
    -------
    pydeck.Viewport
        Viewport fitted to the data
    