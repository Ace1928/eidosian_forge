from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
@property
def specPercent(self):
    if self._spec_targets:
        return six.text_type(_SumPercent(self._spec_targets))
    else:
        return _MISSING_PERCENT_OR_TAGS