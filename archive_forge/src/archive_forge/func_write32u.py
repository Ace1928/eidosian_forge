import struct, sys, time, os
import zlib
import builtins
import io
import _compression
def write32u(output, value):
    output.write(struct.pack('<L', value))