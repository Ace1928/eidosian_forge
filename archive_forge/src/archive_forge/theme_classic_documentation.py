from .elements import element_blank, element_line, element_rect
from .theme import theme
from .theme_bw import theme_bw

    A classic-looking theme, with x & y axis lines and no gridlines

    Parameters
    ----------
    base_size : int
        Base font size. All text sizes are a scaled versions of
        the base font size.
    base_family : str
        Base font family. If `None`, use [](`plotnine.options.base_family`).
    