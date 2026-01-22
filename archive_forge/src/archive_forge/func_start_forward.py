import socket
import ssl
import sys
import tornado.gen
import tornado.httpclient
import tornado.httpserver
import tornado.ioloop
import tornado.iostream
import tornado.web
@tornado.gen.coroutine
def start_forward(reader, writer):
    while True:
        try:
            data = (yield reader.read_bytes(4096, partial=True))
        except tornado.iostream.StreamClosedError:
            break
        if not data:
            break
        writer.write(data)
    writer.close()