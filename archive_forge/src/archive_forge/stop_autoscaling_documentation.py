from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
Stop autoscaling a managed instance group.

    *{command}* stops autoscaling a managed instance group and deletes the
  autoscaler configuration. If autoscaling is not enabled for the managed
  instance group, this command does nothing and will report an error.

  If you need to keep the autoscaler configuration, you can temporarily disable
  an autoscaler by setting its `mode` to `off` using the ``update-autoscaling''
  command instead.

  