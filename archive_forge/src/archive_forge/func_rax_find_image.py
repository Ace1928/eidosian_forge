from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_find_image(module, rax_module, image, exit=True):
    """Find a server image by ID or Name"""
    cs = rax_module.cloudservers
    try:
        UUID(image)
    except ValueError:
        try:
            image = cs.images.find(human_id=image)
        except (cs.exceptions.NotFound, cs.exceptions.NoUniqueMatch):
            try:
                image = cs.images.find(name=image)
            except (cs.exceptions.NotFound, cs.exceptions.NoUniqueMatch):
                if exit:
                    module.fail_json(msg='No matching image found (%s)' % image)
                else:
                    return False
    return rax_module.utils.get_id(image)