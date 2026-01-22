import functools
import matplotlib as _mpl
def set_style(style=None, rc=None):
    """
    Set the aesthetic style of the plots

    This affects things like the color of the axes, whether a grid is
    enabled by default, and other aesthetic elements.

    Parameters
    ----------
    style : "darkgrid" | "whitegrid" | "dark" | "white" | "ticks" | dict | None
        A dictionary of parameters or the name of a preconfigured set.
    rc : dict
        Parameter mappings to override the values in the preset seaborn
        style dictionaries. This only updates parameters that are
        considered part of the style definition.

    Examples
    --------
    >>> set_style("whitegrid")

    >>> set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    See Also
    --------
    axes_style : return a dict of parameters or use in a `with` statement
                 to temporarily set the style.
    set_context : set parameters to scale plot elements
    set_palette : set the default color palette for figures

    """
    style_object = axes_style(style, rc)
    mpl.rcParams.update(style_object)