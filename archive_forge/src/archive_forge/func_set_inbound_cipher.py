import errno
import os
import socket
import struct
import threading
import time
from hmac import HMAC
from paramiko import util
from paramiko.common import (
from paramiko.util import u
from paramiko.ssh_exception import SSHException, ProxyCommandFailure
from paramiko.message import Message
def set_inbound_cipher(self, block_engine, block_size, mac_engine, mac_size, mac_key, etm=False):
    """
        Switch inbound data cipher.
        :param etm: Set encrypt-then-mac from OpenSSH
        """
    self.__block_engine_in = block_engine
    self.__block_size_in = block_size
    self.__mac_engine_in = mac_engine
    self.__mac_size_in = mac_size
    self.__mac_key_in = mac_key
    self.__received_bytes = 0
    self.__received_packets = 0
    self.__received_bytes_overflow = 0
    self.__received_packets_overflow = 0
    self.__etm_in = etm
    self.__init_count |= 2
    if self.__init_count == 3:
        self.__init_count = 0
        self.__need_rekey = False