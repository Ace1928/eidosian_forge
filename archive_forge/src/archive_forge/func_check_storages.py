from django.conf import STATICFILES_STORAGE_ALIAS, settings
from django.contrib.staticfiles.finders import get_finders
from django.core.checks import Error
def check_storages(app_configs=None, **kwargs):
    """Ensure staticfiles is defined in STORAGES setting."""
    errors = []
    if STATICFILES_STORAGE_ALIAS not in settings.STORAGES:
        errors.append(E005)
    return errors