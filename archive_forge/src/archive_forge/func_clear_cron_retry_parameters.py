from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
def clear_cron_retry_parameters(self):
    if self.has_cron_retry_parameters_:
        self.has_cron_retry_parameters_ = 0
        if self.cron_retry_parameters_ is not None:
            self.cron_retry_parameters_.Clear()