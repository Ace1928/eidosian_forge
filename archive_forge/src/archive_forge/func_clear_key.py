from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_key(self):
    if self.has_key_:
        self.has_key_ = 0
        self.key_ = ''