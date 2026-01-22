from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_stringvalue(self):
    if self.has_stringvalue_:
        self.has_stringvalue_ = 0
        self.stringvalue_ = ''