import dataclasses
import socket
import ssl
import threading
import typing as t
def register_encryptor(self, encryptor: MessageEncryptor) -> None:
    self._encryptor = encryptor