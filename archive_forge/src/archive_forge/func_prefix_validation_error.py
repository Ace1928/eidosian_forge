from django.core.exceptions import ValidationError
from django.utils.functional import SimpleLazyObject
from django.utils.text import format_lazy
def prefix_validation_error(error, prefix, code, params):
    """
    Prefix a validation error message while maintaining the existing
    validation data structure.
    """
    if error.error_list == [error]:
        error_params = error.params or {}
        return ValidationError(message=format_lazy('{} {}', SimpleLazyObject(lambda: prefix % params), SimpleLazyObject(lambda: error.message % error_params)), code=code, params={**error_params, **params})
    return ValidationError([prefix_validation_error(e, prefix, code, params) for e in error.error_list])