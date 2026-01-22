from unittest import mock
from keystoneauth1 import plugin
def clear_callstack(self):
    self.client.callstack = []