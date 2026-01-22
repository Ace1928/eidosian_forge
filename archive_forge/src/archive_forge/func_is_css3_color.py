import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
def is_css3_color(instance):
    if instance.lower() in webcolors.css3_names_to_hex:
        return True
    return is_css_color_code(instance)