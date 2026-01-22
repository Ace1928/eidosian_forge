from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
@property
def specTags(self):
    spec_tags = _TAGS_JOIN_STRING.join(sorted((t.tag for t in self._spec_targets if t.tag)))
    return spec_tags if spec_tags else _MISSING_PERCENT_OR_TAGS