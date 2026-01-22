import copy
import json
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.db.models import CASCADE, UUIDField
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch
from django.utils.html import smart_urlquote
from django.utils.http import urlencode
from django.utils.text import Truncator
from django.utils.translation import get_language
from django.utils.translation import gettext as _
def get_select2_language():
    lang_code = get_language()
    supported_code = SELECT2_TRANSLATIONS.get(lang_code)
    if supported_code is None and lang_code is not None:
        i = None
        while (i := lang_code.rfind('-', 0, i)) > -1:
            if (supported_code := SELECT2_TRANSLATIONS.get(lang_code[:i])):
                return supported_code
    return supported_code