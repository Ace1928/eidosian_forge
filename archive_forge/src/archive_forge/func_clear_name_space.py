from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_name_space(self):
    if self.has_name_space_:
        self.has_name_space_ = 0
        self.name_space_ = ''