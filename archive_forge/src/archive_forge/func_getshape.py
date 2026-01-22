import typing
from rpy2.rinterface_lib import openrlib
def getshape(cdata, rk: typing.Optional[int]=None) -> typing.Tuple[int, ...]:
    """Get the shape (size for each dimension) of an R array.

    The rank of the array can optionally by passed. Note that is potentially
    an unsafe operation if the value for the rank in incorrect. It may
    result in a segfault.

    :param cdata: C data from cffi
    :return: A Tuple with the sizes. The length of the tuple is the rank.
    """
    if rk is None:
        rk = getrank(cdata)
    dim_cdata = openrlib.rlib.Rf_getAttrib(cdata, openrlib.rlib.R_DimSymbol)
    shape: typing.Tuple[int, ...]
    if dim_cdata == openrlib.rlib.R_NilValue:
        shape = (openrlib.rlib.Rf_length(cdata),)
    else:
        _ = []
        for i in range(rk):
            _.append(openrlib.INTEGER_ELT(dim_cdata, i))
        shape = tuple(_)
    return shape