from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
def enable_image(module, client, image, enable):
    image = client.image.info(image.ID)
    changed = False
    state = image.STATE
    if state not in [IMAGE_STATES.index('READY'), IMAGE_STATES.index('DISABLED'), IMAGE_STATES.index('ERROR')]:
        if enable:
            module.fail_json(msg='Cannot enable ' + IMAGE_STATES[state] + ' image!')
        else:
            module.fail_json(msg='Cannot disable ' + IMAGE_STATES[state] + ' image!')
    if enable and state != IMAGE_STATES.index('READY') or (not enable and state != IMAGE_STATES.index('DISABLED')):
        changed = True
    if changed and (not module.check_mode):
        client.image.enable(image.ID, enable)
    result = get_image_info(image)
    result['changed'] = changed
    return result