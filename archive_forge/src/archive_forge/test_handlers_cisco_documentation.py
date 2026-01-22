from __future__ import absolute_import, division, print_function
import logging
from passlib import hash, exc
from passlib.utils.compat import u
from .utils import UserHandlerMixin, HandlerCase, repeat_string
from .test_handlers import UPASS_TABLE
test salt value border cases