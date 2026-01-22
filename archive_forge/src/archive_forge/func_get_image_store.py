from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
def get_image_store(self):
    if self.image_store:
        return self.image_store
    image_store = self.module.params.get('name')
    args = {'name': self.module.params.get('name'), 'zoneid': self.get_zone(key='id')}
    image_stores = self.query_api('listImageStores', **args)
    if image_stores:
        for img_s in image_stores.get('imagestore'):
            if image_store.lower() in [img_s['name'].lower(), img_s['id']]:
                self.image_store = img_s
                break
    return self.image_store