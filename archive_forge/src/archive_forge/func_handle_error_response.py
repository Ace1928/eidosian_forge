import abc
import base64
import enum
import json
import six
from google.auth import exceptions
def handle_error_response(response_body):
    """Translates an error response from an OAuth operation into an
    OAuthError exception.

    Args:
        response_body (str): The decoded response data.

    Raises:
        google.auth.exceptions.OAuthError
    """
    try:
        error_components = []
        error_data = json.loads(response_body)
        error_components.append('Error code {}'.format(error_data['error']))
        if 'error_description' in error_data:
            error_components.append(': {}'.format(error_data['error_description']))
        if 'error_uri' in error_data:
            error_components.append(' - {}'.format(error_data['error_uri']))
        error_details = ''.join(error_components)
    except (KeyError, ValueError):
        error_details = response_body
    raise exceptions.OAuthError(error_details, response_body)