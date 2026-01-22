from __future__ import print_function
import socket
import sys
def writeall(sock):
    while True:
        data = sock.recv(256)
        if not data:
            sys.stdout.write('\r\n*** EOF ***\r\n\r\n')
            sys.stdout.flush()
            break
        sys.stdout.write(data)
        sys.stdout.flush()