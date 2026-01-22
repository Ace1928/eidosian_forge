from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.document_renderers import renderer
Renders a table.

    Nested tables are not supported.

    Args:
      table: renderer.TableAttributes object.
      rows: A list of rows, each row is a list of column strings.
    