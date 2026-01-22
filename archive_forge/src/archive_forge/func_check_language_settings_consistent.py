from django.conf import settings
from django.utils.translation import get_supported_language_variant
from django.utils.translation.trans_real import language_code_re
from . import Error, Tags, register
@register(Tags.translation)
def check_language_settings_consistent(app_configs, **kwargs):
    """Error if language settings are not consistent with each other."""
    try:
        get_supported_language_variant(settings.LANGUAGE_CODE)
    except LookupError:
        return [E004]
    else:
        return []