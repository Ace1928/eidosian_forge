from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def get_linux_mounts(module, mntinfo_file='/proc/self/mountinfo'):
    """Gather mount information"""
    try:
        f = open(mntinfo_file)
    except IOError:
        return
    lines = map(str.strip, f.readlines())
    try:
        f.close()
    except IOError:
        module.fail_json(msg='Cannot close file %s' % mntinfo_file)
    mntinfo = {}
    for line in lines:
        fields = line.split()
        record = {'id': int(fields[0]), 'parent_id': int(fields[1]), 'root': fields[3], 'dst': fields[4], 'opts': fields[5], 'fs': fields[-3], 'src': fields[-2]}
        mntinfo[record['id']] = record
    mounts = {}
    for mnt in mntinfo.values():
        if mnt['parent_id'] != 1 and mnt['parent_id'] in mntinfo:
            m = mntinfo[mnt['parent_id']]
            if len(m['root']) > 1 and mnt['root'].startswith('%s/' % m['root']):
                mnt['root'] = mnt['root'][len(m['root']):]
            if m['dst'] != '/':
                mnt['root'] = '%s%s' % (m['dst'], mnt['root'])
            src = mnt['root']
        else:
            src = mnt['src']
        record = {'dst': mnt['dst'], 'src': src, 'opts': mnt['opts'], 'fs': mnt['fs']}
        mounts[mnt['dst']] = record
    return mounts