from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_order_to_v1_order(self, v3_order, v1_order):
    """Converts a v3 Query order to a v1 PropertyOrder.

    Args:
      v3_order: a datastore_pb.Query.Order
      v1_order: a googledatastore.PropertyOrder to populate
    """
    v1_order.property.name = v3_order.property()
    if v3_order.has_direction():
        v1_order.direction = v3_order.direction()