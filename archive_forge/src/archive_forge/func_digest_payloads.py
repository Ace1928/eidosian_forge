from __future__ import print_function
import re
import hashlib
@classmethod
def digest_payloads(cls, msg):
    for part in msg.walk():
        if part.get_content_maintype() == 'text':
            payload = part.get_payload(decode=True)
            charset = part.get_content_charset()
            errors = 'ignore'
            if not charset:
                charset = 'ascii'
            elif charset.lower().replace('_', '-') in ('quopri-codec', 'quopri', 'quoted-printable', 'quotedprintable'):
                errors = 'strict'
            try:
                payload = payload.decode(charset, errors)
            except (LookupError, UnicodeError, AssertionError):
                try:
                    payload = payload.decode('ascii', 'ignore')
                except UnicodeError:
                    continue
            if part.get_content_subtype() == 'html':
                yield cls.normalize_html_part(payload)
            else:
                yield payload
        elif part.is_multipart():
            pass
        else:
            yield part.get_payload()