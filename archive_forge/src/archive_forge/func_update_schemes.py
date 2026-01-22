from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
from passlib.utils.compat import suppress_cause
from passlib.ext.django.utils import DJANGO_VERSION, DjangoTranslator, _PasslibHasherWrapper
from passlib.tests.utils import TestCase, TEST_MODE
from .test_ext_django import (
@receiver(setting_changed, weak=False)
def update_schemes(**kwds):
    if kwds and kwds['setting'] != 'PASSWORD_HASHERS':
        return
    assert context is adapter.context
    schemes = [django_to_passlib_name(import_string(hash_path)()) for hash_path in settings.PASSWORD_HASHERS]
    if 'hex_md5' in schemes and 'django_salted_md5' not in schemes:
        schemes.append('django_salted_md5')
    schemes.append('django_disabled')
    context.update(schemes=schemes, deprecated='auto')
    adapter.reset_hashers()