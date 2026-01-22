import pathlib
from django.conf import settings
from django.core.cache import DEFAULT_CACHE_ALIAS, caches
from django.core.cache.backends.filebased import FileBasedCache
from . import Error, Tags, Warning, register
@register(Tags.caches)
def check_file_based_cache_is_absolute(app_configs, **kwargs):
    errors = []
    for alias, config in settings.CACHES.items():
        cache = caches[alias]
        if not isinstance(cache, FileBasedCache):
            continue
        if not pathlib.Path(config['LOCATION']).is_absolute():
            errors.append(Warning(f"Your '{alias}' cache LOCATION path is relative. Use an absolute path instead.", id='caches.W003'))
    return errors