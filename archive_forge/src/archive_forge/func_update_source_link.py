from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible.module_utils.basic import AnsibleModule
def update_source_link(module, fusion, current, patches):
    source_link = get_source_link_from_parameters(module.params)
    if source_link is not None and (current.source is None or current.source.self_link != source_link):
        patch = purefusion.VolumePatch(source_link=purefusion.NullableString(source_link))
        patches.append(patch)