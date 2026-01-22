from __future__ import (absolute_import, division, print_function)
import os
import re
import tempfile
from ansible.module_utils.six import PY2
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
def read_certificate_fingerprint(self, cert_format='PEM'):
    if self.ssl_backend == 'cryptography':
        if cert_format == 'PEM':
            cert_loader = load_pem_x509_certificate
        else:
            cert_loader = load_der_x509_certificate
        try:
            with open(self.certificate_path, 'rb') as cert_file:
                cert = cert_loader(cert_file.read(), backend=backend)
        except (OSError, ValueError) as e:
            self.module.fail_json(msg='Unable to read the provided certificate: %s' % to_native(e))
        fp = hex_decode(cert.fingerprint(hashes.SHA256())).upper()
        fingerprint = ':'.join([fp[i:i + 2] for i in range(0, len(fp), 2)])
    else:
        current_certificate_fingerprint_cmd = [self.openssl_bin, 'x509', '-noout', '-in', self.certificate_path, '-fingerprint', '-sha256']
        rc, current_certificate_fingerprint_out, current_certificate_fingerprint_err = self.module.run_command(current_certificate_fingerprint_cmd, environ_update=None, check_rc=False)
        if rc != 0:
            return self.module.fail_json(msg=current_certificate_fingerprint_out, err=current_certificate_fingerprint_err, cmd=current_certificate_fingerprint_cmd, rc=rc)
        current_certificate_match = re.search('=([\\w:]+)', current_certificate_fingerprint_out)
        if not current_certificate_match:
            return self.module.fail_json(msg='Unable to find the current certificate fingerprint in %s' % current_certificate_fingerprint_out, cmd=current_certificate_fingerprint_cmd, rc=rc)
        fingerprint = current_certificate_match.group(1)
    return fingerprint