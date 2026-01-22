from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def getcap(self, path):
    rval = []
    cmd = '%s -v %s' % (self.getcap_cmd, path)
    rc, stdout, stderr = self.module.run_command(cmd)
    if rc != 0 or stderr != '':
        self.module.fail_json(msg='Unable to get capabilities of %s' % path, stdout=stdout.strip(), stderr=stderr)
    if stdout.strip() != path:
        if ' =' in stdout:
            caps = stdout.split(' =')[1].strip().split()
        else:
            caps = stdout.split()[1].strip().split()
        for cap in caps:
            cap = cap.lower()
            if ',' in cap:
                cap_group = cap.split(',')
                cap_group[-1], op, flags = self._parse_cap(cap_group[-1])
                for subcap in cap_group:
                    rval.append((subcap, op, flags))
            else:
                rval.append(self._parse_cap(cap))
    return rval