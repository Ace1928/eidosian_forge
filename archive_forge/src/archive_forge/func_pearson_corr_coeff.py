import numpy as np
from scipy.stats import pearsonr
from .._shared.utils import check_shape_equality, as_binary_ndarray
def pearson_corr_coeff(image0, image1, mask=None):
    """Calculate Pearson's Correlation Coefficient between pixel intensities
    in channels.

    Parameters
    ----------
    image0 : (M, N) ndarray
        Image of channel A.
    image1 : (M, N) ndarray
        Image of channel 2 to be correlated with channel B.
        Must have same dimensions as `image0`.
    mask : (M, N) ndarray of dtype bool, optional
        Only `image0` and `image1` pixels within this region of interest mask
        are included in the calculation. Must have same dimensions as `image0`.

    Returns
    -------
    pcc : float
        Pearson's correlation coefficient of the pixel intensities between
        the two images, within the mask if provided.
    p-value : float
        Two-tailed p-value.

    Notes
    -----
    Pearson's Correlation Coefficient (PCC) measures the linear correlation
    between the pixel intensities of the two images. Its value ranges from -1
    for perfect linear anti-correlation to +1 for perfect linear correlation.
    The calculation of the p-value assumes that the intensities of pixels in
    each input image are normally distributed.

    Scipy's implementation of Pearson's correlation coefficient is used. Please
    refer to it for further information and caveats [1]_.

    .. math::
        r = \\frac{\\sum (A_i - m_A_i) (B_i - m_B_i)}
        {\\sqrt{\\sum (A_i - m_A_i)^2 \\sum (B_i - m_B_i)^2}}

    where
        :math:`A_i` is the value of the :math:`i^{th}` pixel in `image0`
        :math:`B_i` is the value of the :math:`i^{th}` pixel in `image1`,
        :math:`m_A_i` is the mean of the pixel values in `image0`
        :math:`m_B_i` is the mean of the pixel values in `image1`

    A low PCC value does not necessarily mean that there is no correlation
    between the two channel intensities, just that there is no linear
    correlation. You may wish to plot the pixel intensities of each of the two
    channels in a 2D scatterplot and use Spearman's rank correlation if a
    non-linear correlation is visually identified [2]_. Also consider if you
    are interested in correlation or co-occurence, in which case a method
    involving segmentation masks (e.g. MCC or intersection coefficient) may be
    more suitable [3]_ [4]_.

    Providing the mask of only relevant sections of the image (e.g., cells, or
    particular cellular compartments) and removing noise is important as the
    PCC is sensitive to these measures [3]_ [4]_.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html  # noqa
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html  # noqa
    .. [3] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical
           guide to evaluating colocalization in biological microscopy.
           American journal of physiology. Cell physiology, 300(4), C723–C742.
           https://doi.org/10.1152/ajpcell.00462.2010
    .. [4] Bolte, S. and Cordelières, F.P. (2006), A guided tour into
           subcellular colocalization analysis in light microscopy. Journal of
           Microscopy, 224: 213-232.
           https://doi.org/10.1111/j.1365-2818.2006.01706.x
    """
    image0 = np.asarray(image0)
    image1 = np.asarray(image1)
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name='mask')
        check_shape_equality(image0, image1, mask)
        image0 = image0[mask]
        image1 = image1[mask]
    else:
        check_shape_equality(image0, image1)
        image0 = image0.reshape(-1)
        image1 = image1.reshape(-1)
    return tuple((float(v) for v in pearsonr(image0, image1)))