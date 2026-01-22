from ..sage_helper import _within_sage
import math

    Unfortunately, the short_slopes_from_translations uses the convention
    that the longitude translation is real whereas the SnapPea kernel and
    M.cusp_translations uses the convention that the meridian translation
    is real. Thus, we have a flag here.

    Ideally, we would rewrite below short_slopes_from_translations to
    use the same convention.
    