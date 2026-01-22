from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_user_organization(self):
    if self.has_user_organization_:
        self.has_user_organization_ = 0
        self.user_organization_ = ''