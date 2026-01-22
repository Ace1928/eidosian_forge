from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_referrer_policy(app_configs, **kwargs):
    if _security_middleware():
        if settings.SECURE_REFERRER_POLICY is None:
            return [W022]
        if isinstance(settings.SECURE_REFERRER_POLICY, str):
            values = {v.strip() for v in settings.SECURE_REFERRER_POLICY.split(',')}
        else:
            values = set(settings.SECURE_REFERRER_POLICY)
        if not values <= REFERRER_POLICY_VALUES:
            return [E023]
    return []