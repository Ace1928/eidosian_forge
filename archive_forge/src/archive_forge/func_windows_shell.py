from __future__ import print_function
import socket
import sys
def windows_shell(chan):
    import threading
    sys.stdout.write('Line-buffered terminal emulation. Press F6 or ^Z to send EOF.\r\n\r\n')

    def writeall(sock):
        while True:
            data = sock.recv(256)
            if not data:
                sys.stdout.write('\r\n*** EOF ***\r\n\r\n')
                sys.stdout.flush()
                break
            sys.stdout.write(data)
            sys.stdout.flush()
    writer = threading.Thread(target=writeall, args=(chan,))
    writer.start()
    try:
        while True:
            d = sys.stdin.read(1)
            if not d:
                break
            chan.send(d)
    except EOFError:
        pass