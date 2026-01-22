import collections
import weakref
from tensorflow.python.trackable import base
from tensorflow.python.trackable import converter
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
Returns a list of all nodes and its paths from self.root using a breadth first traversal.