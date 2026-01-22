from . import exceptions
from . import misc
from . import normalizers
def ensure_one_of(allowed_values, uri, attribute):
    """Assert that the uri's attribute is one of the allowed values."""
    value = getattr(uri, attribute)
    if value is not None and allowed_values and (value not in allowed_values):
        raise exceptions.UnpermittedComponentError(attribute, value, allowed_values)