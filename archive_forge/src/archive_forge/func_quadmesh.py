import itertools
from collections import defaultdict
import param
from ..converter import HoloViewsConverter
from ..util import is_list_like, process_dynamic_args
def quadmesh(self, x=None, y=None, z=None, colorbar=True, **kwds):
    """
        QuadMesh plot

        `quadmesh` allows you to plot values on an irregular grid by representing each value as a
        polygon.

        Reference: https://hvplot.holoviz.org/reference/xarray/quadmesh.html

        Parameters
        ----------
        x : string, optional
            The coordinate variable along the x-axis
        y : string, optional
            The coordinate variable along the y-axis
        z : string, optional
            The data variable to plot
        colorbar: boolean
            Whether to display a colorbar
        **kwds : optional
            Additional keywords arguments are documented in `hvplot.help('quadmesh')`.

        Returns
        -------
        A Holoviews object. You can `print` the object to study its composition and run

        .. code-block::

            import holoviews as hv
            hv.help(the_holoviews_object)

        to learn more about its parameters and options.

        Examples
        --------

        .. code-block::

            import hvplot.xarray
            import xarray as xr

            ds = xr.tutorial.open_dataset('rasm')
            ds.Tair.hvplot.quadmesh(x='xc', y='yc', geo=True, widget_location='bottom')

        References
        ----------

        - HoloViews: https://holoviews.org/reference/elements/bokeh/QuadMesh.html
        """
    return self(x, y, z=z, kind='quadmesh', colorbar=colorbar, **kwds)