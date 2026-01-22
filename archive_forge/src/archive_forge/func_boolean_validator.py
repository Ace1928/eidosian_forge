from keystone.common import validation
from keystone.i18n import _
def boolean_validator(value):
    if value not in (True, False):
        raise TypeError(_('Expected boolean value, got %r') % type(value))