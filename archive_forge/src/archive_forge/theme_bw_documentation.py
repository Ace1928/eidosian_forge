from .elements import element_line, element_rect, element_text
from .theme import theme
from .theme_gray import theme_gray

    White background with black gridlines

    Parameters
    ----------
    base_size : int
        Base font size. All text sizes are a scaled versions of
        the base font size.
    base_family : str
        Base font family. If `None`, use [](`plotnine.options.base_family`).
    