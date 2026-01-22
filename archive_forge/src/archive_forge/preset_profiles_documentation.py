from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import exceptions
Parses the profile name string into the corresponding API X509Parameters.

  Args:
    profile_name: The preset profile name.

  Returns:
    An X509Parameters object.
  