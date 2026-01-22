from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_security_middleware(app_configs, **kwargs):
    passed_check = _security_middleware()
    return [] if passed_check else [W001]