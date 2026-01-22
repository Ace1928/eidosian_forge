from collections import Counter
from django.conf import settings
from . import Error, Tags, Warning, register
@register(Tags.urls)
def check_url_namespaces_unique(app_configs, **kwargs):
    """
    Warn if URL namespaces used in applications aren't unique.
    """
    if not getattr(settings, 'ROOT_URLCONF', None):
        return []
    from django.urls import get_resolver
    resolver = get_resolver()
    all_namespaces = _load_all_namespaces(resolver)
    counter = Counter(all_namespaces)
    non_unique_namespaces = [n for n, count in counter.items() if count > 1]
    errors = []
    for namespace in non_unique_namespaces:
        errors.append(Warning("URL namespace '{}' isn't unique. You may not be able to reverse all URLs in this namespace".format(namespace), id='urls.W005'))
    return errors