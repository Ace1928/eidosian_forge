from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_kind(self):
    if self.has_kind_:
        self.has_kind_ = 0
        self.kind_ = 0