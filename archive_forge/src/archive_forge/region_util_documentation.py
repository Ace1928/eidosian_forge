from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.ml_engine import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
Gets the region and prompt for region if not provided.

  Note: region can be either `global` or one of supported regions.

    Region is decided in the following order:
  - region argument;
  - ai_platform/region gcloud config;
  - prompt user input.

  Args:
    args: Namespace, The args namespace.

  Returns:
    A str representing region.
  