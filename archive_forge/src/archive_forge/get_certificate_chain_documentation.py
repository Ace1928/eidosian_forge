from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
Get a certificate chain for a given version.

  Returns the PEM-format certificate chain for the specified key version.
  The optional flag `output-file` indicates the path to store the PEM. If not
  specified, the PEM will be printed to stdout.
  