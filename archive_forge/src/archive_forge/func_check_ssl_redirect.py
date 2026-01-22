from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_ssl_redirect(app_configs, **kwargs):
    passed_check = not _security_middleware() or settings.SECURE_SSL_REDIRECT is True
    return [] if passed_check else [W008]