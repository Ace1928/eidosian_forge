import json
import netaddr
import re
def decode_free_output(value):
    """The value of the output action can be found free, i.e: without the
    'output' keyword. This decoder decodes its value when found this way."""
    try:
        return ('output', {'port': int(value)})
    except ValueError:
        return ('output', {'port': value.strip('"')})