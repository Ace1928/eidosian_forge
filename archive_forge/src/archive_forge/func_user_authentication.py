import importlib.util
import types
import asyncio
import logging
from typing import Any, Optional
@StandardDecorator()
def user_authentication(username: str, password: str) -> bool:
    """
    Authenticates a user based on username and password.

    Args:
        username (str): The user's username.
        password (str): The user's password.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
    if username == 'admin' and password == 'admin':
        logging.info('User authenticated successfully.')
        return True
    logging.warning('Authentication failed.')
    return False