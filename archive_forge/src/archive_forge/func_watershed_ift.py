import numpy
import numpy as np
from . import _ni_support
from . import _ni_label
from . import _nd_image
from . import _morphology
def watershed_ift(input, markers, structure=None, output=None):
    """
    Apply watershed from markers using image foresting transform algorithm.

    Parameters
    ----------
    input : array_like
        Input.
    markers : array_like
        Markers are points within each watershed that form the beginning
        of the process. Negative markers are considered background markers
        which are processed after the other markers.
    structure : structure element, optional
        A structuring element defining the connectivity of the object can be
        provided. If None, an element is generated with a squared
        connectivity equal to one.
    output : ndarray, optional
        An output array can optionally be provided. The same shape as input.

    Returns
    -------
    watershed_ift : ndarray
        Output.  Same shape as `input`.

    References
    ----------
    .. [1] A.X. Falcao, J. Stolfi and R. de Alencar Lotufo, "The image
           foresting transform: theory, algorithms, and applications",
           Pattern Analysis and Machine Intelligence, vol. 26, pp. 19-29, 2004.

    """
    input = numpy.asarray(input)
    if input.dtype.type not in [numpy.uint8, numpy.uint16]:
        raise TypeError('only 8 and 16 unsigned inputs are supported')
    if structure is None:
        structure = _morphology.generate_binary_structure(input.ndim, 1)
    structure = numpy.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for ii in structure.shape:
        if ii != 3:
            raise RuntimeError('structure dimensions must be equal to 3')
    if not structure.flags.contiguous:
        structure = structure.copy()
    markers = numpy.asarray(markers)
    if input.shape != markers.shape:
        raise RuntimeError('input and markers must have equal shape')
    integral_types = [numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.intc, numpy.intp]
    if markers.dtype.type not in integral_types:
        raise RuntimeError('marker should be of integer type')
    if isinstance(output, numpy.ndarray):
        if output.dtype.type not in integral_types:
            raise RuntimeError('output should be of integer type')
    else:
        output = markers.dtype
    output = _ni_support._get_output(output, input)
    _nd_image.watershed_ift(input, markers, structure, output)
    return output