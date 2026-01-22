from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_id(self):
    if self.has_id_:
        self.has_id_ = 0
        self.id_ = 0