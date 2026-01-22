import heapq
import itertools
import logging
import os
import re
import sys
import threading  # Knowing full well that this is a usually a placeholder.
import traceback
from xml.sax import saxutils
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import capabilities
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_query
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
def pb_to_index(self, pb):
    index_def = pb.definition()
    properties = [(property.name().decode('utf-8'), DatastoreAdapter.index_direction_mappings.get(property.direction())) for property in index_def.property_list()]
    index = Index(pb.id(), index_def.entity_type().decode('utf-8'), index_def.ancestor(), properties)
    state = DatastoreAdapter.index_state_mappings.get(pb.state())
    return (index, state)