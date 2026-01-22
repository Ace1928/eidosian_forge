from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
Set autohealing policy for managed instance group.

    *{command}* updates the autohealing policy for an existing managed
  instance group.

  If health check is specified, the resulting autohealing policy will be
  triggered by the health-check signal i.e. the autohealing action (RECREATE) on
  an instance will be performed if the health-check signals that the instance is
  UNHEALTHY. If no health check is specified, the resulting autohealing policy
  will be triggered by instance's status i.e. the autohealing action (RECREATE)
  on an instance will be performed if the instance.status is not RUNNING.
  