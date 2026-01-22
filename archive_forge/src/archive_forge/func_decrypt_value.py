import configparser
import json
from typing import Any, Dict, Optional, Union
import os
import logging
from logging_config import configure_logging
from encryption_key_manager import get_valid_encryption_key
def decrypt_value(self, value: str) -> str:
    """
        Decrypts a configuration value using the configured cipher suite.

        Args:
            value (str): The encrypted value to decrypt.

        Returns:
            str: The decrypted value.
        """
    decrypted_value = self.cipher_suite.decrypt(value.encode()).decode()
    logging.debug(f'Value decrypted.')
    return decrypted_value