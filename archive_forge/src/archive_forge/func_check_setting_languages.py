from django.conf import settings
from django.utils.translation import get_supported_language_variant
from django.utils.translation.trans_real import language_code_re
from . import Error, Tags, register
@register(Tags.translation)
def check_setting_languages(app_configs, **kwargs):
    """Error if LANGUAGES setting is invalid."""
    return [Error(E002.msg.format(tag), id=E002.id) for tag, _ in settings.LANGUAGES if not isinstance(tag, str) or not language_code_re.match(tag)]