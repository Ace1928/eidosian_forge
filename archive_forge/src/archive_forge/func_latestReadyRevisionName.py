from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kuberun import structuredout
from googlecloudsdk.command_lib.kuberun import kubernetes_consts as k8s
@property
def latestReadyRevisionName(self):
    return self._props.get(k8s.FIELD_LATEST_READY_REVISION_NAME)