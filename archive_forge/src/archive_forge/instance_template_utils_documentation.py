from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import alias_ip_range_utils
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnet_flags
from googlecloudsdk.core import properties
Returns a list of accelerator config messages for Instance Templates.

  Args:
    messages: tracked GCE API messages.
    accelerator: accelerator object with the following properties:
        * type: the accelerator's type.
        * count: the number of accelerators to attach. Optional, defaults to 1.

  Returns:
    a list of accelerator config messages that specify the type and number of
    accelerators to attach to an instance.
  