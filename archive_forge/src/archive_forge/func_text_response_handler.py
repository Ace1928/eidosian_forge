import threading
import socket
import select
def text_response_handler(sock):
    request_content = consume_socket_content(sock, timeout=request_timeout)
    sock.send(text.encode('utf-8'))
    return request_content