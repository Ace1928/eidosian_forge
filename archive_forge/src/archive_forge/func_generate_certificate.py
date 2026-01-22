from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
def generate_certificate(self):
    """(Re-)Generate certificate."""
    command = [self.acme_tiny_path]
    if self.use_chain:
        command.append('--chain')
    command.extend(['--account-key', self.accountkey_path])
    if self.csr_content is not None:
        fd, tmpsrc = tempfile.mkstemp()
        self.module.add_cleanup_file(tmpsrc)
        f = os.fdopen(fd, 'wb')
        try:
            f.write(self.csr_content)
        except Exception as err:
            try:
                f.close()
            except Exception as dummy:
                pass
            self.module.fail_json(msg='failed to create temporary CSR file: %s' % to_native(err), exception=traceback.format_exc())
        f.close()
        command.extend(['--csr', tmpsrc])
    else:
        command.extend(['--csr', self.csr_path])
    command.extend(['--acme-dir', self.challenge_path])
    command.extend(['--directory-url', self.acme_directory])
    try:
        self.cert = to_bytes(self.module.run_command(command, check_rc=True)[1])
    except OSError as exc:
        raise CertificateError(exc)