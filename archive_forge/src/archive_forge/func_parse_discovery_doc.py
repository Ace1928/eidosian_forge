import json
import logging
from typing import Dict, NamedTuple, Optional, Union
import urllib
from absl import flags
from utils import bq_consts
from utils import bq_error
def parse_discovery_doc(discovery_document: Union[str, bytes]) -> Dict[str, str]:
    """Takes a downloaded discovery document and parses it.

  Args:
    discovery_document: The discovery doc to parse.

  Returns:
    The parsed api doc.
  """
    if isinstance(discovery_document, str):
        return json.loads(discovery_document)
    elif isinstance(discovery_document, bytes):
        return json.loads(discovery_document.decode('utf-8'))
    raise ValueError(f'Unsupported discovery document type: {type(discovery_document)}')