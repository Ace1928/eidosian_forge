from urllib.parse import unquote, urlsplit, urlunsplit
from asgiref.local import Local
from django.utils.functional import lazy
from django.utils.translation import override
from .exceptions import NoReverseMatch, Resolver404
from .resolvers import _get_cached_resolver, get_ns_resolver, get_resolver
from .utils import get_callable
def get_urlconf(default=None):
    """
    Return the root URLconf to use for the current thread if it has been
    changed from the default one.
    """
    return getattr(_urlconfs, 'value', default)