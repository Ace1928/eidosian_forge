from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
def check_image_id(compute_api, image_id):
    response = compute_api.get(path='images/%s' % image_id)
    if not response.ok:
        msg = 'Error in getting image %s on %s : %s' % (image_id, compute_api.module.params.get('api_url'), response.json)
        compute_api.module.fail_json(msg=msg)