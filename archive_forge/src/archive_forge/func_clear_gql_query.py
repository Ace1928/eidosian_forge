from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_gql_query(self):
    if self.has_gql_query_:
        self.has_gql_query_ = 0
        if self.gql_query_ is not None:
            self.gql_query_.Clear()