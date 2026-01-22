from binascii import hexlify
import configparser
from contextlib import contextmanager
from fcntl import ioctl
import os
import struct
import uuid
from os_brick import exception
from os_brick import privileged
@privileged.default.entrypoint
def get_connector_password(filename, config_group, failed_over):
    """Read ScaleIO connector configuration file and get appropriate password.

    :param filename: path to connector configuration file
    :type filename: str
    :param config_group: name of section in configuration file
    :type config_group: str
    :param failed_over: flag representing if storage is in failed over state
    :type failed_over: bool
    :return: connector password
    :rtype: str
    """
    if not os.path.isfile(filename):
        msg = 'ScaleIO connector configuration file is not found in path %s.' % filename
        raise exception.BrickException(message=msg)
    conf = configparser.ConfigParser()
    conf.read(filename)
    password_key = 'replicating_san_password' if failed_over else 'san_password'
    return conf[config_group][password_key]