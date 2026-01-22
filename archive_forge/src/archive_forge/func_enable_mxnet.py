import warnings
from packaging.version import Version
def enable_mxnet():
    warn_msg = 'Built-in MXNet support will be removed in Thinc v9. If you need MXNet support in the future, you can transition to using a custom copy of the current MXNetWrapper in your package or project.'
    warnings.warn(warn_msg, DeprecationWarning)
    global mxnet, has_mxnet
    import mxnet
    has_mxnet = True