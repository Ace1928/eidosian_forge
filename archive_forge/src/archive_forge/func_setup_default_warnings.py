import warnings
from .compat import Literal
def setup_default_warnings():
    filter_warning('ignore', error_msg='numpy.dtype size changed')
    filter_warning('ignore', error_msg='numpy.ufunc size changed')
    for pipe in ['matcher', 'entity_ruler', 'span_ruler']:
        filter_warning('once', error_msg=Warnings.W036.format(name=pipe))
    filter_warning('once', error_msg=Warnings.W108)
    filter_warning('once', error_msg='[W114]')