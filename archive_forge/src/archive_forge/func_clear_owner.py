from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_owner(self):
    if self.has_owner_:
        self.has_owner_ = 0
        if self.owner_ is not None:
            self.owner_.Clear()