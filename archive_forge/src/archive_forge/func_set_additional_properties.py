from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Any, Dict
def set_additional_properties(message, additional_properties: Dict[Any, Any]):
    """Sets additional properties on a message."""
    ls = [message.AdditionalProperty(key=key, value=value) for key, value in additional_properties.items()]
    message.additionalProperties = ls
    return message