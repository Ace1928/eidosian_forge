from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
Capture a screenshot (JPEG image) of the virtual machine instance's display.