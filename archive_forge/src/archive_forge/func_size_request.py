import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def size_request(widget):

    class SizeRequest(UserList):

        def __init__(self, req):
            self.height = req.height
            self.width = req.width
            UserList.__init__(self, [self.width, self.height])
    return SizeRequest(orig_size_request(widget))