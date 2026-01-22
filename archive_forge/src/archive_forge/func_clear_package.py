from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_package(self):
    if self.has_package_:
        self.has_package_ = 0
        self.package_ = ''