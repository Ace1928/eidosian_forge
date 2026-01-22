from __future__ import absolute_import, division, print_function
import binascii
import os
import re
from time import sleep
from datetime import datetime
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils._text import to_native
def certificate_fingerprint(self, path):
    """Load x509 certificate that is either encoded DER or PEM encoding and return the certificate fingerprint."""
    certificate = None
    with open(path, 'rb') as fh:
        data = fh.read()
        try:
            certificate = x509.load_pem_x509_certificate(data, default_backend())
        except Exception as error:
            try:
                certificate = x509.load_der_x509_certificate(data, default_backend())
            except Exception as error:
                self.module.fail_json(msg='Failed to determine certificate fingerprint. File [%s]. Array [%s]. Error [%s].' % (path, self.ssid, to_native(error)))
    return binascii.hexlify(certificate.fingerprint(certificate.signature_hash_algorithm)).decode('utf-8')