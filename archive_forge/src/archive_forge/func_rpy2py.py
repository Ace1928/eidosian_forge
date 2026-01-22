from .. import utils
from .._lazyload import anndata2ri
from .._lazyload import rpy2
import numpy as np
import warnings
@utils._with_pkg(pkg='rpy2', min_version='3.0')
def rpy2py(robject):
    """Convert an rpy2 object to Python.

    Attempts the following, in order: data.frame -> pd.DataFrame, named list -> dict,
    unnamed list -> list, SingleCellExperiment -> anndata.AnnData, vector -> np.ndarray,
    rpy2 generic converter, NULL -> None.

    Parameters
    ----------
    robject : rpy2 object
        Object to be converted

    Returns
    -------
    pyobject : python object
        Converted object
    """
    for converter in [_rpynull2py, _rpysce2py, rpy2.robjects.pandas2ri.converter.rpy2py, _rpylist2py, rpy2.robjects.numpy2ri.converter.rpy2py, rpy2.robjects.default_converter.rpy2py]:
        if _is_r_object(robject):
            try:
                robject = converter(robject)
            except NotImplementedError:
                pass
        else:
            break
    if _is_r_object(robject):
        warnings.warn('Object not converted: {} (type {})'.format(robject, type(robject).__name__), RuntimeWarning)
    return robject