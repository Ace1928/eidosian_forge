import base64
import binascii
import collections
import html
from django.conf import settings
from django.core.exceptions import (
from django.core.files.uploadhandler import SkipFile, StopFutureHandlers, StopUpload
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.http import parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile

        Find a multipart boundary in data.

        Should no boundary exist in the data, return None. Otherwise, return
        a tuple containing the indices of the following:
         * the end of current encapsulation
         * the start of the next encapsulation
        