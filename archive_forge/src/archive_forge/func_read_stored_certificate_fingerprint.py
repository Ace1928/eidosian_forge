from __future__ import (absolute_import, division, print_function)
import os
import re
import tempfile
from ansible.module_utils.six import PY2
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
def read_stored_certificate_fingerprint(self):
    stored_certificate_fingerprint_cmd = [self.keytool_bin, '-list', '-alias', self.name, '-keystore', self.keystore_path, '-v']
    rc, stored_certificate_fingerprint_out, stored_certificate_fingerprint_err = self.module.run_command(stored_certificate_fingerprint_cmd, data=self.password, check_rc=False)
    if rc != 0:
        if 'keytool error: java.lang.Exception: Alias <%s> does not exist' % self.name in stored_certificate_fingerprint_out:
            return 'alias mismatch'
        if re.match('keytool error: java\\.io\\.IOException: ' + '[Kk]eystore( was tampered with, or)? password was incorrect', stored_certificate_fingerprint_out):
            return 'password mismatch'
        return self.module.fail_json(msg=stored_certificate_fingerprint_out, err=stored_certificate_fingerprint_err, cmd=stored_certificate_fingerprint_cmd, rc=rc)
    if self.keystore_type not in (None, self.current_type()):
        return 'keystore type mismatch'
    stored_certificate_match = re.search('SHA256: ([\\w:]+)', stored_certificate_fingerprint_out)
    if not stored_certificate_match:
        return self.module.fail_json(msg='Unable to find the stored certificate fingerprint in %s' % stored_certificate_fingerprint_out, cmd=stored_certificate_fingerprint_cmd, rc=rc)
    return stored_certificate_match.group(1)