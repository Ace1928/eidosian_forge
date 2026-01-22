from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def manage_sources(self):
    force = self.params['force']
    source = self.params['source']
    imgtype = self.params['type']
    cmd = '{0} sources'.format(self.cmd)
    if force:
        cmd += ' -f'
    if self.present:
        cmd = '{0} -a {1} -t {2}'.format(cmd, source, imgtype)
        rc, stdout, stderr = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Failed to add source: {0}'.format(self.errmsg(stderr)))
        regex = 'Already have "{0}" image source "{1}", no change'.format(imgtype, source)
        if re.match(regex, stdout):
            self.changed = False
        regex = 'Added "%s" image source "%s"' % (imgtype, source)
        if re.match(regex, stdout):
            self.changed = True
    else:
        cmd += ' -d %s' % source
        rc, stdout, stderr = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Failed to remove source: {0}'.format(self.errmsg(stderr)))
        regex = 'Do not have image source "%s", no change' % source
        if re.match(regex, stdout):
            self.changed = False
        regex = 'Deleted ".*" image source "%s"' % source
        if re.match(regex, stdout):
            self.changed = True