from __future__ import absolute_import, unicode_literals
from oauthlib.common import quote, unicode_type, unquote
def filter_params(target):
    """Decorator which filters params to remove non-oauth_* parameters

    Assumes the decorated method takes a params dict or list of tuples as its
    first argument.
    """

    def wrapper(params, *args, **kwargs):
        params = filter_oauth_params(params)
        return target(params, *args, **kwargs)
    wrapper.__doc__ = target.__doc__
    return wrapper