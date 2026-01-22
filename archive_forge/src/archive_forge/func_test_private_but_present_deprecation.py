import pkgutil
import types
import importlib
import warnings
from importlib import import_module
import pytest
import scipy
@pytest.mark.parametrize(('module_name', 'correct_module'), [('scipy.constants.codata', None), ('scipy.constants.constants', None), ('scipy.fftpack.basic', None), ('scipy.fftpack.helper', None), ('scipy.fftpack.pseudo_diffs', None), ('scipy.fftpack.realtransforms', None), ('scipy.integrate.dop', None), ('scipy.integrate.lsoda', None), ('scipy.integrate.odepack', None), ('scipy.integrate.quadpack', None), ('scipy.integrate.vode', None), ('scipy.interpolate.fitpack', None), ('scipy.interpolate.fitpack2', None), ('scipy.interpolate.interpolate', None), ('scipy.interpolate.ndgriddata', None), ('scipy.interpolate.polyint', None), ('scipy.interpolate.rbf', None), ('scipy.io.harwell_boeing', None), ('scipy.io.idl', None), ('scipy.io.mmio', None), ('scipy.io.netcdf', None), ('scipy.io.arff.arffread', 'arff'), ('scipy.io.matlab.byteordercodes', 'matlab'), ('scipy.io.matlab.mio_utils', 'matlab'), ('scipy.io.matlab.mio', 'matlab'), ('scipy.io.matlab.mio4', 'matlab'), ('scipy.io.matlab.mio5_params', 'matlab'), ('scipy.io.matlab.mio5_utils', 'matlab'), ('scipy.io.matlab.mio5', 'matlab'), ('scipy.io.matlab.miobase', 'matlab'), ('scipy.io.matlab.streams', 'matlab'), ('scipy.linalg.basic', None), ('scipy.linalg.decomp', None), ('scipy.linalg.decomp_cholesky', None), ('scipy.linalg.decomp_lu', None), ('scipy.linalg.decomp_qr', None), ('scipy.linalg.decomp_schur', None), ('scipy.linalg.decomp_svd', None), ('scipy.linalg.flinalg', None), ('scipy.linalg.matfuncs', None), ('scipy.linalg.misc', None), ('scipy.linalg.special_matrices', None), ('scipy.misc.common', None), ('scipy.ndimage.filters', None), ('scipy.ndimage.fourier', None), ('scipy.ndimage.interpolation', None), ('scipy.ndimage.measurements', None), ('scipy.ndimage.morphology', None), ('scipy.odr.models', None), ('scipy.odr.odrpack', None), ('scipy.optimize.cobyla', None), ('scipy.optimize.lbfgsb', None), ('scipy.optimize.linesearch', None), ('scipy.optimize.minpack', None), ('scipy.optimize.minpack2', None), ('scipy.optimize.moduleTNC', None), ('scipy.optimize.nonlin', None), ('scipy.optimize.optimize', None), ('scipy.optimize.slsqp', None), ('scipy.optimize.tnc', None), ('scipy.optimize.zeros', None), ('scipy.signal.bsplines', None), ('scipy.signal.filter_design', None), ('scipy.signal.fir_filter_design', None), ('scipy.signal.lti_conversion', None), ('scipy.signal.ltisys', None), ('scipy.signal.signaltools', None), ('scipy.signal.spectral', None), ('scipy.signal.waveforms', None), ('scipy.signal.wavelets', None), ('scipy.signal.windows.windows', 'windows'), ('scipy.sparse.lil', None), ('scipy.sparse.linalg.dsolve', 'linalg'), ('scipy.sparse.linalg.eigen', 'linalg'), ('scipy.sparse.linalg.interface', 'linalg'), ('scipy.sparse.linalg.isolve', 'linalg'), ('scipy.sparse.linalg.matfuncs', 'linalg'), ('scipy.sparse.sparsetools', None), ('scipy.sparse.spfuncs', None), ('scipy.sparse.sputils', None), ('scipy.spatial.ckdtree', None), ('scipy.spatial.kdtree', None), ('scipy.spatial.qhull', None), ('scipy.spatial.transform.rotation', 'transform'), ('scipy.special.add_newdocs', None), ('scipy.special.basic', None), ('scipy.special.orthogonal', None), ('scipy.special.sf_error', None), ('scipy.special.specfun', None), ('scipy.special.spfun_stats', None), ('scipy.stats.biasedurn', None), ('scipy.stats.kde', None), ('scipy.stats.morestats', None), ('scipy.stats.mstats_basic', 'mstats'), ('scipy.stats.mstats_extras', 'mstats'), ('scipy.stats.mvn', None), ('scipy.stats.stats', None)])
def test_private_but_present_deprecation(module_name, correct_module):
    module = import_module(module_name)
    if correct_module is None:
        import_name = f'scipy.{module_name.split('.')[1]}'
    else:
        import_name = f'scipy.{module_name.split('.')[1]}.{correct_module}'
    correct_import = import_module(import_name)
    for attr_name in module.__all__:
        attr = getattr(correct_import, attr_name, None)
        if attr is None:
            message = f'`{module_name}.{attr_name}` is deprecated...'
        else:
            message = f'Please import `{attr_name}` from the `{import_name}`...'
        with pytest.deprecated_call(match=message):
            getattr(module, attr_name)
    message = f'`{module_name}` is deprecated...'
    with pytest.raises(AttributeError, match=message):
        getattr(module, 'ekki')