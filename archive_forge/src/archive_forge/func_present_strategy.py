from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
from ansible.module_utils.basic import AnsibleModule
from uuid import uuid4
def present_strategy(api, security_group):
    ret = {'changed': False}
    response = api.get('security_groups')
    if not response.ok:
        api.module.fail_json(msg='Error getting security groups "%s": "%s" (%s)' % (response.info['msg'], response.json['message'], response.json))
    security_group_lookup = dict(((sg['name'], sg) for sg in response.json['security_groups']))
    if security_group['name'] not in security_group_lookup.keys():
        ret['changed'] = True
        if api.module.check_mode:
            ret['scaleway_security_group'] = {'id': str(uuid4())}
            return ret
        response = api.post('/security_groups', data=payload_from_security_group(security_group))
        if not response.ok:
            msg = 'Error during security group creation: "%s": "%s" (%s)' % (response.info['msg'], response.json['message'], response.json)
            api.module.fail_json(msg=msg)
        ret['scaleway_security_group'] = response.json['security_group']
    else:
        ret['scaleway_security_group'] = security_group_lookup[security_group['name']]
    return ret