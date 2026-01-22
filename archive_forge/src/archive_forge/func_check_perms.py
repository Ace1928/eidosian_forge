from functools import wraps
from urllib.parse import urlparse
from django.conf import settings
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.core.exceptions import PermissionDenied
from django.shortcuts import resolve_url
def check_perms(user):
    if isinstance(perm, str):
        perms = (perm,)
    else:
        perms = perm
    if user.has_perms(perms):
        return True
    if raise_exception:
        raise PermissionDenied
    return False