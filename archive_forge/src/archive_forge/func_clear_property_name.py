from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_property_name(self):
    if self.has_property_name_:
        self.has_property_name_ = 0
        self.property_name_ = ''