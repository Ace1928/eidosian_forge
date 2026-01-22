from urllib.parse import urlparse
from django.conf import settings
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.shortcuts import resolve_url
def handle_no_permission(self):
    if self.raise_exception or self.request.user.is_authenticated:
        raise PermissionDenied(self.get_permission_denied_message())
    path = self.request.build_absolute_uri()
    resolved_login_url = resolve_url(self.get_login_url())
    login_scheme, login_netloc = urlparse(resolved_login_url)[:2]
    current_scheme, current_netloc = urlparse(path)[:2]
    if (not login_scheme or login_scheme == current_scheme) and (not login_netloc or login_netloc == current_netloc):
        path = self.request.get_full_path()
    return redirect_to_login(path, resolved_login_url, self.get_redirect_field_name())