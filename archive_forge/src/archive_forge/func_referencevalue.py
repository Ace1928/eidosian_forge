from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def referencevalue(self):
    if self.referencevalue_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.referencevalue_ is None:
                self.referencevalue_ = PropertyValue_ReferenceValue()
        finally:
            self.lazy_init_lock_.release()
    return self.referencevalue_