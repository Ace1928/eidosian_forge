import json
import logging
import os
import socket
from threading import RLock
from filelock import FileLock
from ray.autoscaler._private.local.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def record_local_head_state_if_needed(local_provider: LocalNodeProvider) -> None:
    """This function is called on the Ray head from StandardAutoscaler.reset
    to record the head node's own existence in the cluster state file.

    This is necessary because `provider.create_node` in
    `commands.get_or_create_head_node` records the head state on the
    cluster-launching machine but not on the head.
    """
    head_ip = local_provider.provider_config['head_ip']
    cluster_name = local_provider.cluster_name
    if head_ip not in local_provider.non_terminated_nodes({}):
        head_tags = {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_USER_NODE_TYPE: LOCAL_CLUSTER_NODE_TYPE, TAG_RAY_NODE_NAME: 'ray-{}-head'.format(cluster_name), TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}
        local_provider.create_node(node_config={}, tags=head_tags, count=1)
        assert head_ip in local_provider.non_terminated_nodes({})