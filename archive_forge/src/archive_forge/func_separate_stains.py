from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def separate_stains(rgb, conv_matrix, *, channel_axis=-1):
    """RGB to stain color space conversion.

    Parameters
    ----------
    rgb : (..., C=3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    conv_matrix: ndarray
        The stain separation matrix as described by G. Landini [1]_.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in stain color space. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    Stain separation matrices available in the ``color`` module and their
    respective colorspace:

    * ``hed_from_rgb``: Hematoxylin + Eosin + DAB
    * ``hdx_from_rgb``: Hematoxylin + DAB
    * ``fgx_from_rgb``: Feulgen + Light Green
    * ``bex_from_rgb``: Giemsa stain : Methyl Blue + Eosin
    * ``rbd_from_rgb``: FastRed + FastBlue +  DAB
    * ``gdx_from_rgb``: Methyl Green + DAB
    * ``hax_from_rgb``: Hematoxylin + AEC
    * ``bro_from_rgb``: Blue matrix Anilline Blue + Red matrix Azocarmine                        + Orange matrix Orange-G
    * ``bpx_from_rgb``: Methyl Blue + Ponceau Fuchsin
    * ``ahx_from_rgb``: Alcian Blue + Hematoxylin
    * ``hpx_from_rgb``: Hematoxylin + PAS

    This implementation borrows some ideas from DIPlib [2]_, e.g. the
    compensation using a small value to avoid log artifacts when
    calculating the Beer-Lambert law.

    References
    ----------
    .. [1] https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html
    .. [2] https://github.com/DIPlib/diplib/
    .. [3] A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical
           staining by color deconvolution,” Anal. Quant. Cytol. Histol., vol.
           23, no. 4, pp. 291–299, Aug. 2001.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import separate_stains, hdx_from_rgb
    >>> ihc = data.immunohistochemistry()
    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    """
    rgb = _prepare_colorarray(rgb, force_copy=True, channel_axis=-1)
    np.maximum(rgb, 1e-06, out=rgb)
    log_adjust = np.log(1e-06)
    stains = np.log(rgb) / log_adjust @ conv_matrix
    np.maximum(stains, 0, out=stains)
    return stains