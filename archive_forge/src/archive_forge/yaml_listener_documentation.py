from __future__ import absolute_import
import copy
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
Call YAML parser to generate and handle all events.

    Calls PyYAML parser and sends resulting generator to handle_event method
    for processing.

    Args:
      stream: String document or open file object to process as per the
        yaml.parse method.  Any object that implements a 'read()' method which
        returns a string document will work with the YAML parser.
      loader_class: Used for dependency injection.
      **loader_args: Pass to the loader on construction.
    