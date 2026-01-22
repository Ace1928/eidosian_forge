import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def enable_mpl_offline(resize=False, strip_style=False, verbose=False, show_link=False, link_text='Export to plot.ly', validate=True):
    """
    Convert mpl plots to locally hosted HTML documents.

    This function should be used with the inline matplotlib backend
    that ships with IPython that can be enabled with `%pylab inline`
    or `%matplotlib inline`. This works by adding an HTML formatter
    for Figure objects; the existing SVG/PNG formatters will remain
    enabled.

    (idea taken from `mpld3._display.enable_notebook`)

    Example:
    ```
    from plotly.offline import enable_mpl_offline
    import matplotlib.pyplot as plt

    enable_mpl_offline()

    fig = plt.figure()
    x = [10, 15, 20, 25, 30]
    y = [100, 250, 200, 150, 300]
    plt.plot(x, y, "o")
    fig
    ```
    """
    init_notebook_mode()
    ipython = get_module('IPython')
    matplotlib = get_module('matplotlib')
    ip = ipython.core.getipython.get_ipython()
    formatter = ip.display_formatter.formatters['text/html']
    formatter.for_type(matplotlib.figure.Figure, lambda fig: iplot_mpl(fig, resize, strip_style, verbose, show_link, link_text, validate))