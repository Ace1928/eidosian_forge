from ..options import get_option
from .elements import element_blank, element_line, element_rect, element_text
from .theme import theme

    Theme for seaborn.

    Credit to Michael Waskom's seaborn:

        - http://stanford.edu/~mwaskom/software/seaborn
        - https://github.com/mwaskom/seaborn

    Parameters
    ----------
    style: "whitegrid", "darkgrid", "nogrid", "ticks"
        Style of axis background.
    context: "notebook", "talk", "paper", "poster"]``
        Intended context for resulting figures.
    font : str
        Font family, see matplotlib font manager.
    font_scale : float
        Separate scaling factor to independently scale the
        size of the font elements.
    