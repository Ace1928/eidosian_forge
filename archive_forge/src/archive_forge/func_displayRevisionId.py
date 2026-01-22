from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
@property
def displayRevisionId(self):
    """Returns human readable revision identifier."""
    if self.latestRevision:
        return '%s (currently %s)' % (traffic.GetKey(self), self.revisionName)
    else:
        return self.revisionName