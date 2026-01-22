from __future__ import absolute_import, unicode_literals
from oauthlib.common import extract_params, urlencode
from . import utils
def prepare_form_encoded_body(oauth_params, body):
    """Prepare the Form-Encoded Body.

    Per `section 3.5.2`_ of the spec.

    .. _`section 3.5.2`: https://tools.ietf.org/html/rfc5849#section-3.5.2

    """
    return _append_params(oauth_params, body)