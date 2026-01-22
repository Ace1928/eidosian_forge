from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb
def postfix_position(self):
    if self.postfix_position_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.postfix_position_ is None:
                self.postfix_position_ = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.IndexPostfix()
        finally:
            self.lazy_init_lock_.release()
    return self.postfix_position_