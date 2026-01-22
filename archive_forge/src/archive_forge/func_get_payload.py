import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def get_payload(self, i=None, decode=False):
    """Return a reference to the payload.

        The payload will either be a list object or a string.  If you mutate
        the list object, you modify the message's payload in place.  Optional
        i returns that index into the payload.

        Optional decode is a flag indicating whether the payload should be
        decoded or not, according to the Content-Transfer-Encoding header
        (default is False).

        When True and the message is not a multipart, the payload will be
        decoded if this header's value is `quoted-printable' or `base64'.  If
        some other encoding is used, or the header is missing, or if the
        payload has bogus data (i.e. bogus base64 or uuencoded data), the
        payload is returned as-is.

        If the message is a multipart and the decode flag is True, then None
        is returned.
        """
    if self.is_multipart():
        if decode:
            return None
        if i is None:
            return self._payload
        else:
            return self._payload[i]
    if i is not None and (not isinstance(self._payload, list)):
        raise TypeError('Expected list, got %s' % type(self._payload))
    payload = self._payload
    cte = str(self.get('content-transfer-encoding', '')).lower()
    if not decode:
        if isinstance(payload, str) and utils._has_surrogates(payload):
            try:
                bpayload = payload.encode('ascii', 'surrogateescape')
                try:
                    payload = bpayload.decode(self.get_param('charset', 'ascii'), 'replace')
                except LookupError:
                    payload = bpayload.decode('ascii', 'replace')
            except UnicodeEncodeError:
                pass
        return payload
    if isinstance(payload, str):
        try:
            bpayload = payload.encode('ascii', 'surrogateescape')
        except UnicodeEncodeError:
            bpayload = payload.encode('raw-unicode-escape')
    if cte == 'quoted-printable':
        return quopri.decodestring(bpayload)
    elif cte == 'base64':
        value, defects = decode_b(b''.join(bpayload.splitlines()))
        for defect in defects:
            self.policy.handle_defect(self, defect)
        return value
    elif cte in ('x-uuencode', 'uuencode', 'uue', 'x-uue'):
        try:
            return _decode_uu(bpayload)
        except ValueError:
            return bpayload
    if isinstance(payload, str):
        return bpayload
    return payload