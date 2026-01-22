import re
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.core.mail import mail_managers
from django.http import HttpResponsePermanentRedirect
from django.urls import is_valid_path
from django.utils.deprecation import MiddlewareMixin
from django.utils.http import escape_leading_slashes
def should_redirect_with_slash(self, request):
    """
        Return True if settings.APPEND_SLASH is True and appending a slash to
        the request path turns an invalid path into a valid one.
        """
    if settings.APPEND_SLASH and (not request.path_info.endswith('/')):
        urlconf = getattr(request, 'urlconf', None)
        if not is_valid_path(request.path_info, urlconf):
            match = is_valid_path('%s/' % request.path_info, urlconf)
            if match:
                view = match.func
                return getattr(view, 'should_append_slash', True)
    return False