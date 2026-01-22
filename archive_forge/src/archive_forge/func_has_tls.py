import logging
import os
import sys
def has_tls() -> bool:
    try:
        import _ssl
        return True
    except ImportError:
        pass
    from pip._vendor.urllib3.util import IS_PYOPENSSL
    return IS_PYOPENSSL