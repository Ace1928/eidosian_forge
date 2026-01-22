from typing import Dict, Optional, Union
from .. import constants
from ._runtime import (
from ._token import get_token
from ._validators import validate_hf_hub_args
@validate_hf_hub_args
def build_hf_headers(*, token: Optional[Union[bool, str]]=None, is_write_action: bool=False, library_name: Optional[str]=None, library_version: Optional[str]=None, user_agent: Union[Dict, str, None]=None) -> Dict[str, str]:
    """
    Build headers dictionary to send in a HF Hub call.

    By default, authorization token is always provided either from argument (explicit
    use) or retrieved from the cache (implicit use). To explicitly avoid sending the
    token to the Hub, set `token=False` or set the `HF_HUB_DISABLE_IMPLICIT_TOKEN`
    environment variable.

    In case of an API call that requires write access, an error is thrown if token is
    `None` or token is an organization token (starting with `"api_org***"`).

    In addition to the auth header, a user-agent is added to provide information about
    the installed packages (versions of python, huggingface_hub, torch, tensorflow,
    fastai and fastcore).

    Args:
        token (`str`, `bool`, *optional*):
            The token to be sent in authorization header for the Hub call:
                - if a string, it is used as the Hugging Face token
                - if `True`, the token is read from the machine (cache or env variable)
                - if `False`, authorization header is not set
                - if `None`, the token is read from the machine only except if
                  `HF_HUB_DISABLE_IMPLICIT_TOKEN` env variable is set.
        is_write_action (`bool`, default to `False`):
            Set to True if the API call requires a write access. If `True`, the token
            will be validated (cannot be `None`, cannot start by `"api_org***"`).
        library_name (`str`, *optional*):
            The name of the library that is making the HTTP request. Will be added to
            the user-agent header.
        library_version (`str`, *optional*):
            The version of the library that is making the HTTP request. Will be added
            to the user-agent header.
        user_agent (`str`, `dict`, *optional*):
            The user agent info in the form of a dictionary or a single string. It will
            be completed with information about the installed packages.

    Returns:
        A `Dict` of headers to pass in your API call.

    Example:
    ```py
        >>> build_hf_headers(token="hf_***") # explicit token
        {"authorization": "Bearer hf_***", "user-agent": ""}

        >>> build_hf_headers(token=True) # explicitly use cached token
        {"authorization": "Bearer hf_***",...}

        >>> build_hf_headers(token=False) # explicitly don't use cached token
        {"user-agent": ...}

        >>> build_hf_headers() # implicit use of the cached token
        {"authorization": "Bearer hf_***",...}

        # HF_HUB_DISABLE_IMPLICIT_TOKEN=True # to set as env variable
        >>> build_hf_headers() # token is not sent
        {"user-agent": ...}

        >>> build_hf_headers(token="api_org_***", is_write_action=True)
        ValueError: You must use your personal account token for write-access methods.

        >>> build_hf_headers(library_name="transformers", library_version="1.2.3")
        {"authorization": ..., "user-agent": "transformers/1.2.3; hf_hub/0.10.2; python/3.10.4; tensorflow/1.55"}
    ```

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If organization token is passed and "write" access is required.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If "write" access is required but token is not passed and not saved locally.
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If `token=True` but token is not saved locally.
    """
    token_to_send = get_token_to_send(token)
    _validate_token_to_send(token_to_send, is_write_action=is_write_action)
    headers = {'user-agent': _http_user_agent(library_name=library_name, library_version=library_version, user_agent=user_agent)}
    if token_to_send is not None:
        headers['authorization'] = f'Bearer {token_to_send}'
    return headers