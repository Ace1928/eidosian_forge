import numpy as np
from .._shared.utils import _supported_float_type
from .colorconv import lab2lch, _cart2polar_2pi
def deltaE_cmc(lab1, lab2, kL=1, kC=1, *, channel_axis=-1):
    """Color difference from the  CMC l:c standard.

    This color difference was developed by the Colour Measurement Committee
    (CMC) of the Society of Dyers and Colourists (United Kingdom). It is
    intended for use in the textile industry.

    The scale factors `kL`, `kC` set the weight given to differences in
    lightness and chroma relative to differences in hue.  The usual values are
    ``kL=2``, ``kC=1`` for "acceptability" and ``kL=1``, ``kC=1`` for
    "imperceptibility".  Colors with ``dE > 1`` are "different" for the given
    scale factors.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    Notes
    -----
    deltaE_cmc the defines the scales for the lightness, hue, and chroma
    in terms of the first color.  Consequently
    ``deltaE_cmc(lab1, lab2) != deltaE_cmc(lab2, lab1)``

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    .. [3] F. J. J. Clarke, R. McDonald, and B. Rigg, "Modification to the
           JPC79 colour-difference formula," J. Soc. Dyers Colour. 100, 128-132
           (1984).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    lab1 = np.moveaxis(lab1, source=channel_axis, destination=0)
    lab2 = np.moveaxis(lab2, source=channel_axis, destination=0)
    L1, C1, h1 = lab2lch(lab1, channel_axis=0)[:3]
    L2, C2, h2 = lab2lch(lab2, channel_axis=0)[:3]
    dC = C1 - C2
    dL = L1 - L2
    dH2 = get_dH2(lab1, lab2, channel_axis=0)
    T = np.where(np.logical_and(np.rad2deg(h1) >= 164, np.rad2deg(h1) <= 345), 0.56 + 0.2 * np.abs(np.cos(h1 + np.deg2rad(168))), 0.36 + 0.4 * np.abs(np.cos(h1 + np.deg2rad(35))))
    c1_4 = C1 ** 4
    F = np.sqrt(c1_4 / (c1_4 + 1900))
    SL = np.where(L1 < 16, 0.511, 0.040975 * L1 / (1.0 + 0.01765 * L1))
    SC = 0.638 + 0.0638 * C1 / (1.0 + 0.0131 * C1)
    SH = SC * (F * T + 1 - F)
    dE2 = (dL / (kL * SL)) ** 2
    dE2 += (dC / (kC * SC)) ** 2
    dE2 += dH2 / SH ** 2
    return np.sqrt(np.maximum(dE2, 0))