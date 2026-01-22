from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def check_example_section_errors(self):
    """Raise violation if the examples section does not contain a valid example.

    Also, wrap up the examples section by specifying there are no errors in the
    section.

    See go/cloud-sdk-help-text#formatting.
    """
    if self.needs_example() and (not self.example):
        self._add_failure(self._check_name('EXAMPLES', 'PRESENT'), 'You have not included an example in the Examples section.')
    elif self._has_example_section and (not self._example_errors):
        self._add_no_errors_summary('EXAMPLES')
    self.example = True