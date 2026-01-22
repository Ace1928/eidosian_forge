from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
Generates the detailed help doc for a command.

  Args:
    resource: The name of the resource. e.g "instance", "image" or "disk"
    brief_doc_template: The brief doc template to use.
    description_template: The command description template.
    example_template: The example template to use.
  Returns:
    The detailed help doc for a command. The returned value is a map with
    two attributes; 'brief' and 'description'.
  