from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
@abc.abstractmethod
def parameter_numbers(self) -> DHParameterNumbers:
    """
        Returns a DHParameterNumbers.
        """