from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_cross_origin_opener_policy(app_configs, **kwargs):
    if _security_middleware() and settings.SECURE_CROSS_ORIGIN_OPENER_POLICY is not None and (settings.SECURE_CROSS_ORIGIN_OPENER_POLICY not in CROSS_ORIGIN_OPENER_POLICY_VALUES):
        return [E024]
    return []