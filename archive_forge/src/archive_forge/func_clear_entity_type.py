from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_entity_type(self):
    if self.has_entity_type_:
        self.has_entity_type_ = 0
        self.entity_type_ = ''