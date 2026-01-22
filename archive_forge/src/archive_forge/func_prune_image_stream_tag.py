from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_images_common import (
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def prune_image_stream_tag(self, stream, tag_event_list):
    manifests_to_delete, images_to_delete = ([], [])
    filtered_items = []
    tag_event_items = tag_event_list['items'] or []
    prune_over_size_limit = self.params.get('prune_over_size_limit')
    stream_namespace = stream['metadata']['namespace']
    stream_name = stream['metadata']['name']
    for idx, item in enumerate(tag_event_items):
        if is_created_after(item['created'], self.max_creation_timestamp):
            filtered_items.append(item)
            continue
        if idx == 0:
            istag = '%s/%s:%s' % (stream_namespace, stream_name, tag_event_list['tag'])
            if istag in self.used_tags:
                filtered_items.append(item)
                continue
        if item['image'] not in self.image_mapping:
            continue
        image = self.image_mapping[item['image']]
        if prune_over_size_limit and (not self.exceeds_limits(stream_namespace, image)):
            filtered_items.append(item)
            continue
        image_ref = '%s/%s@%s' % (stream_namespace, stream_name, item['image'])
        if image_ref in self.used_images:
            filtered_items.append(item)
            continue
        images_to_delete.append(item['image'])
        if self.params.get('prune_registry'):
            manifests_to_delete.append(image['metadata']['name'])
            path = stream_namespace + '/' + stream_name
            image_blobs, err = get_image_blobs(image)
            if not err:
                self.delete_layers_links(path, image_blobs)
    return (filtered_items, manifests_to_delete, images_to_delete)