import itertools
from django.conf import settings
from django.middleware.csrf import get_token
from django.utils.functional import SimpleLazyObject, lazy
def csrf(request):
    """
    Context processor that provides a CSRF token, or the string 'NOTPROVIDED' if
    it has not been provided by either a view decorator or the middleware
    """

    def _get_val():
        token = get_token(request)
        if token is None:
            return 'NOTPROVIDED'
        else:
            return token
    return {'csrf_token': SimpleLazyObject(_get_val)}