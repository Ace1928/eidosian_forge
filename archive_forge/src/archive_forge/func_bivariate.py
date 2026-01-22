import itertools
from collections import defaultdict
import param
from ..converter import HoloViewsConverter
from ..util import is_list_like, process_dynamic_args
def bivariate(self, x=None, y=None, colorbar=True, **kwds):
    """
        A bivariate, density plot uses nested contours (or contours plus colors) to indicate
        regions of higher local density.

        `bivariate` plots can be a useful alternative to scatter plots, if your data are too dense
        to plot each point individually.

        Reference: https://hvplot.holoviz.org/reference/tabular/bivariate.html

        Parameters
        ----------
        x : string, optional
            Field name to draw x-positions from. If not specified, the index is used.
        y : string, optional
            Field name to draw y-positions from
        colorbar: boolean
            Whether to display a colorbar
        bandwidth: int, optional
            The bandwidth of the kernel for the density estimate. Default is None.
        cut: Integer, Optional
            Draw the estimate to cut * bw from the extreme data points. Default is None.
        filled : bool, optional
            If True the the contours will be filled. Default is False.
        levels: int, optional
            The number of contour lines to draw. Default is 10.

        **kwds : optional
            Additional keywords arguments are documented in `hvplot.help('bivariate')`.

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

            import hvplot.pandas
            from bokeh.sampledata.autompg import autompg_clean as df

            bivariate = df.hvplot.bivariate("accel", "mpg", filled=True, cmap="blues")
            bivariate

        To get a better intuitive understanding of the `bivariate` plot, you can try overlaying the
        corresponding scatter plot.

        .. code-block::

            scatter = df.hvplot.scatter("accel", "mpg")
            bivariate * scatter

        References
        ----------

        - ggplot: https://bio304-class.github.io/bio304-fall2017/ggplot-bivariate.html
        - HoloViews: https://holoviews.org/reference/elements/bokeh/Bivariate.html
        - Plotly: https://plotly.com/python/2d-histogram-contour/
        - Matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html
        - Seaborn: https://seaborn.pydata.org/generated/seaborn.kdeplot.html
        - Wiki: https://en.wikipedia.org/wiki/Bivariate_analysis
        """
    return self(x, y, kind='bivariate', colorbar=colorbar, **kwds)