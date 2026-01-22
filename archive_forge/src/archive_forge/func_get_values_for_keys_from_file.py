from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import boto3
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
from six.moves import configparser
def get_values_for_keys_from_file(file_path, keys):
    """Reads JSON or INI file and returns dict with values for requested keys.

  JSON file keys should be top level.
  INI file sections will be flattened.

  Args:
    file_path (str): Path of JSON or INI file to read.
    keys (list[str]): Search for these keys to return from file.

  Returns:
    Dict[cred_key: cred_value].

  Raises:
    ValueError: The file was the incorrect format.
    KeyError: Duplicate key found.
  """
    result = {}
    real_path = os.path.realpath(os.path.expanduser(file_path))
    with files.FileReader(real_path) as file_reader:
        try:
            file_dict = json.loads(file_reader.read())
            _extract_keys(keys, file_dict, result)
        except json.JSONDecodeError:
            config = configparser.ConfigParser()
            try:
                config.read(real_path)
            except configparser.ParsingError:
                raise ValueError('Source creds file must be JSON or INI format.')
            for section in config:
                section_dict = dict(config[section])
                _extract_keys(keys, section_dict, result)
    return result