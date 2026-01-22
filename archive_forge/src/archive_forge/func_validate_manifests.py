from __future__ import absolute_import, division, print_function
import xml.etree.ElementTree as ET
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def validate_manifests(self):
    url = 'https://' + self.esxi_hostname + '/cgi-bin/vm-support.cgi?listmanifests=1'
    headers = self.generate_req_headers(url)
    manifests = []
    try:
        resp, info = fetch_url(self.module, method='GET', headers=headers, url=url)
        if info['status'] != 200:
            self.module.fail_json(msg='failed to fetch manifests from %s: %s' % (url, info['msg']))
        manifest_list = ET.fromstring(resp.read())
        for manifest in manifest_list[0]:
            manifests.append(manifest.attrib['id'])
    except Exception as e:
        self.module.fail_json(msg='Failed to fetch manifests from %s: %s' % (url, e))
    for manifest in self.manifests:
        validate_manifest_result = [m for m in manifests if m == manifest]
        if not validate_manifest_result:
            self.module.fail_json(msg='%s is a manifest that cannot be specified.' % manifest)