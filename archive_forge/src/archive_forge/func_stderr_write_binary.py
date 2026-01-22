import os
import sys
def stderr_write_binary(data):
    encoding = sys.stderr.encoding or sys.getdefaultencoding()
    sys.stderr.write(data.decode(encoding))