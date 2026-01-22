from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import pem_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.util import files
Creates a CA pool object and update mask from CA pool update flags.

  Requires that args has 'publish-crl', 'publish-ca-cert', and
  update labels flags registered.

  Args:
    args: The parser that contains the flag values.
    current_labels: The current set of labels for the CA pool.

  Returns:
    A tuple with the CA pool object to update with and the list of strings
    representing the update mask, respectively.
  