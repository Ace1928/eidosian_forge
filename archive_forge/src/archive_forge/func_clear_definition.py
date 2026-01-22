from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_definition(self):
    self.has_definition_ = 0
    self.definition_.Clear()