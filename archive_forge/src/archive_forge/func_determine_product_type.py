from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.events import iam_util
from googlecloudsdk.api_lib.kuberun.core import events_constants
from googlecloudsdk.command_lib.events import exceptions
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def determine_product_type(client, authentication):
    """Determine eventing product type inferred by namespaces."""
    product_type = _fetch_product_type(client)
    if product_type == events_constants.Product.CLOUDRUN and authentication == events_constants.AUTH_WI_GSA:
        raise exceptions.UnsupportedArgumentError('This cluster version does not support using Cloud Run events with workload identity.')
    return product_type