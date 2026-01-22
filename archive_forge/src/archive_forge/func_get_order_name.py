from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.transfer.appliances import regions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def get_order_name(locations_id, orders_id):
    """Returns an appliance name to locations and orders ID."""
    return resources.Resource.RelativeName(resources.REGISTRY.Create(ORDERS_COLLECTION, ordersId=orders_id, locationsId=locations_id, projectsId=properties.VALUES.core.project.Get()))