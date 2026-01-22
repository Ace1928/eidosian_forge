import logging
import os
from os_ken.lib import ip
def wrap_and_handle_ssl(sock, addr):
    handle(ssl.wrap_socket(sock, **ssl_args), addr)