import builtins
import datetime
import decimal
from http import client as http_client
import pytz
import re
def is_valid_code(code_value):
    """
    This function checks if incoming value in http response codes range.
    """
    return code_value in http_client.responses