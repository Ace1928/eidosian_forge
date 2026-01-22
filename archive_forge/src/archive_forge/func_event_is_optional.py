from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import enum
@property
def event_is_optional(self):
    return self.provider.default_event == self