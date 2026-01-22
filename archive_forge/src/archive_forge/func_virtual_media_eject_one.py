from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
def virtual_media_eject_one(self, image_url):
    response = self.get_request(self.root_uri + self.systems_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'VirtualMedia' not in data:
        response = self.get_request(self.root_uri + self.manager_uri)
        if response['ret'] is False:
            return response
        data = response['data']
        if 'VirtualMedia' not in data:
            return {'ret': False, 'msg': 'VirtualMedia resource not found'}
    virt_media_uri = data['VirtualMedia']['@odata.id']
    response = self.get_request(self.root_uri + virt_media_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    virt_media_list = []
    for member in data[u'Members']:
        virt_media_list.append(member[u'@odata.id'])
    resources, headers = self._read_virt_media_resources(virt_media_list)
    uri, data, eject = self._find_virt_media_to_eject(resources, image_url)
    if uri and eject:
        if 'Actions' not in data or '#VirtualMedia.EjectMedia' not in data['Actions']:
            h = headers[uri]
            if 'allow' in h:
                methods = [m.strip() for m in h.get('allow').split(',')]
                if 'PATCH' not in methods:
                    return {'ret': False, 'msg': '%s action not found and PATCH not allowed' % '#VirtualMedia.EjectMedia'}
            return self.virtual_media_eject_via_patch(uri)
        else:
            action = data['Actions']['#VirtualMedia.EjectMedia']
            if 'target' not in action:
                return {'ret': False, 'msg': 'target URI property missing from Action #VirtualMedia.EjectMedia'}
            action_uri = action['target']
            payload = {}
            response = self.post_request(self.root_uri + action_uri, payload)
            if response['ret'] is False:
                return response
            return {'ret': True, 'changed': True, 'msg': 'VirtualMedia ejected'}
    elif uri and (not eject):
        return {'ret': True, 'changed': False, 'msg': "VirtualMedia image '%s' already ejected" % image_url}
    else:
        return {'ret': False, 'changed': False, 'msg': "No VirtualMedia resource found with image '%s' inserted" % image_url}