import io
import math
import os
import typing
import weakref
def getColorHSV(name: str) -> tuple:
    """Retrieve the hue, saturation, value triple of a color name.

    Returns:
        a triple (degree, percent, percent). If not found (-1, -1, -1) is returned.
    """
    try:
        x = getColorInfoList()[getColorList().index(name.upper())]
    except Exception:
        if g_exceptions_verbose:
            fitz.exception_info()
        return (-1, -1, -1)
    r = x[1] / 255.0
    g = x[2] / 255.0
    b = x[3] / 255.0
    cmax = max(r, g, b)
    V = round(cmax * 100, 1)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta == 0:
        hue = 0
    elif cmax == r:
        hue = 60.0 * ((g - b) / delta % 6)
    elif cmax == g:
        hue = 60.0 * ((b - r) / delta + 2)
    else:
        hue = 60.0 * ((r - g) / delta + 4)
    H = int(round(hue))
    if cmax == 0:
        sat = 0
    else:
        sat = delta / cmax
    S = int(round(sat * 100))
    return (H, S, V)