import re
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.core.mail import mail_managers
from django.http import HttpResponsePermanentRedirect
from django.urls import is_valid_path
from django.utils.deprecation import MiddlewareMixin
from django.utils.http import escape_leading_slashes
def get_full_path_with_slash(self, request):
    """
        Return the full path of the request with a trailing slash appended.

        Raise a RuntimeError if settings.DEBUG is True and request.method is
        POST, PUT, or PATCH.
        """
    new_path = request.get_full_path(force_append_slash=True)
    new_path = escape_leading_slashes(new_path)
    if settings.DEBUG and request.method in ('POST', 'PUT', 'PATCH'):
        raise RuntimeError("You called this URL via %(method)s, but the URL doesn't end in a slash and you have APPEND_SLASH set. Django can't redirect to the slash URL while maintaining %(method)s data. Change your form to point to %(url)s (note the trailing slash), or set APPEND_SLASH=False in your Django settings." % {'method': request.method, 'url': request.get_host() + new_path})
    return new_path