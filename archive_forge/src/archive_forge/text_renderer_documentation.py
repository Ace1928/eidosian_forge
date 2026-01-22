from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
Renders NAME and SYNOPSIS lines as a second line indent.

    Collapses adjacent spaces to one space, deletes trailing space, and doesn't
    split top-level nested [...] or (...) groups. Also detects and does not
    count terminal control sequences.

    Args:
      line: The NAME or SYNOPSIS text.
      is_synopsis: if it is the synopsis section
    