from warnings import warn
from .utils import STRING_TYPE, logger, NO_VALUE
@property
def puppet(self):
    warn('UnexpectedToken.puppet attribute has been renamed to interactive_parser', DeprecationWarning)
    return self.interactive_parser