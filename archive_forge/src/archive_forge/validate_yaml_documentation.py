from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
Validate a YAML file against a JSON Schema.

  {command} validates YAML / JSON files against
  [JSON Schemas](https://json-schema.org/).
  