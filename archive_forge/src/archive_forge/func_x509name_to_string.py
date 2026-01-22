from typing import Any, Optional
import OpenSSL._util as pyOpenSSLutil
import OpenSSL.SSL
import OpenSSL.version
from OpenSSL.crypto import X509Name
from scrapy.utils.python import to_unicode
def x509name_to_string(x509name: X509Name) -> str:
    result_buffer: Any = pyOpenSSLutil.ffi.new('char[]', 512)
    pyOpenSSLutil.lib.X509_NAME_oneline(x509name._name, result_buffer, len(result_buffer))
    return ffi_buf_to_string(result_buffer)