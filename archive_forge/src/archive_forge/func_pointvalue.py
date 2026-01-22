from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def pointvalue(self):
    if self.pointvalue_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.pointvalue_ is None:
                self.pointvalue_ = PropertyValue_PointValue()
        finally:
            self.lazy_init_lock_.release()
    return self.pointvalue_