import functools
import matplotlib as _mpl
def set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None):
    """
    Set aesthetic parameters in one step

    Each set of parameters can be set directly or temporarily, see the
    referenced functions below for more information.

    Parameters
    ----------
    context : string or dict
        Plotting context parameters, see :func:`plotting_context`
    style : string or dict
        Axes style parameters, see :func:`axes_style`
    palette : string or sequence
        Color palette, see :func:`color_palette`
    font : string
        Font family, see matplotlib font manager.
    font_scale : float
        Separate scaling factor to independently scale the size of the
        font elements.
    color_codes : bool
        If `True` and `palette` is a seaborn palette, remap the shorthand
        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.
    rc : dict or None
        Dictionary of rc parameter mappings to override the above.

    """
    set_context(context, font_scale)
    set_style(style, rc={'font.family': font})
    if rc is not None:
        mpl.rcParams.update(rc)
    return mpl.rcParams