from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.core import log
List the Cloud Composer image version upgrades for a specific environment.

  {command} prints a table listing the suggested image-version upgrades with the
  following columns:
  * Image Version ID
  * Composer 'default' flag
  * List of supported python versions
  