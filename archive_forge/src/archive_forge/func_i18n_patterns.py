import functools
from django.conf import settings
from django.urls import LocalePrefixPattern, URLResolver, get_resolver, path
from django.views.i18n import set_language
def i18n_patterns(*urls, prefix_default_language=True):
    """
    Add the language code prefix to every URL pattern within this function.
    This may only be used in the root URLconf, not in an included URLconf.
    """
    if not settings.USE_I18N:
        return list(urls)
    return [URLResolver(LocalePrefixPattern(prefix_default_language=prefix_default_language), list(urls))]