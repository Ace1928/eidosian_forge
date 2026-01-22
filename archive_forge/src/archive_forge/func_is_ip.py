import socket
import ssl
import socks # $ pip install PySocks
def is_ip(s):
    try:
        if ':' in s:
            socket.inet_pton(socket.AF_INET6, s)
        elif '.' in s:
            socket.inet_aton(s)
        else:
            return False
    except:
        return False
    else:
        return True