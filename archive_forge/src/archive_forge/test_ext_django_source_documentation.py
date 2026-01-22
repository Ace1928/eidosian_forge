from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
from passlib.utils.compat import suppress_cause
from passlib.ext.django.utils import DJANGO_VERSION, DjangoTranslator, _PasslibHasherWrapper
from passlib.tests.utils import TestCase, TEST_MODE
from .test_ext_django import (

                wrapper around arbitrary func, that first triggers sync
                