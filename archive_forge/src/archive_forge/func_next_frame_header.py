import errno
import os
import select
import socket as pysocket
import struct
def next_frame_header(socket):
    """
    Returns the stream and size of the next frame of data waiting to be read
    from socket, according to the protocol defined here:

    https://docs.docker.com/engine/api/v1.24/#attach-to-a-container
    """
    try:
        data = read_exactly(socket, 8)
    except SocketError:
        return (-1, -1)
    stream, actual = struct.unpack('>BxxxL', data)
    return (stream, actual)