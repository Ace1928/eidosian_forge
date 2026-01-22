import codecs
import copy
from io import BytesIO
from itertools import chain
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit
from django.conf import settings
from django.core import signing
from django.core.exceptions import (
from django.core.files import uploadhandler
from django.http.multipartparser import (
from django.utils.datastructures import (
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain, parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
def parse_file_upload(self, META, post_data):
    """Return a tuple of (POST QueryDict, FILES MultiValueDict)."""
    self.upload_handlers = ImmutableList(self.upload_handlers, warning='You cannot alter upload handlers after the upload has been processed.')
    parser = MultiPartParser(META, post_data, self.upload_handlers, self.encoding)
    return parser.parse()