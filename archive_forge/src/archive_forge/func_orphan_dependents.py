from pprint import pformat
from six import iteritems
import re
@orphan_dependents.setter
def orphan_dependents(self, orphan_dependents):
    """
        Sets the orphan_dependents of this V1DeleteOptions.
        Deprecated: please use the PropagationPolicy, this field will be
        deprecated in 1.7. Should the dependent objects be orphaned. If
        true/false, the "orphan" finalizer will be added to/removed from the
        object's finalizers list. Either this field or PropagationPolicy may be
        set, but not both.

        :param orphan_dependents: The orphan_dependents of this V1DeleteOptions.
        :type: bool
        """
    self._orphan_dependents = orphan_dependents