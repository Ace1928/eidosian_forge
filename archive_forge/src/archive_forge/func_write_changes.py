from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.module_utils.basic import AnsibleModule
def write_changes(module, contents, path):
    tmpfd, tmpfile = tempfile.mkstemp(dir=module.tmpdir)
    f = os.fdopen(tmpfd, 'wb')
    f.write(contents)
    f.close()
    validate = module.params.get('validate', None)
    valid = not validate
    if validate:
        if '%s' not in validate:
            module.fail_json(msg='validate must contain %%s: %s' % validate)
        rc, out, err = module.run_command(validate % tmpfile)
        valid = rc == 0
        if rc != 0:
            module.fail_json(msg='failed to validate: rc:%s error:%s' % (rc, err))
    if valid:
        module.atomic_move(tmpfile, path, unsafe_writes=module.params['unsafe_writes'])