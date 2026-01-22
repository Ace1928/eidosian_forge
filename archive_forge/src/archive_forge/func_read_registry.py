import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def read_registry(self):
    """Extract resolver configuration from the Windows registry."""
    lm = _winreg.ConnectRegistry(None, _winreg.HKEY_LOCAL_MACHINE)
    want_scan = False
    try:
        try:
            tcp_params = _winreg.OpenKey(lm, 'SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters')
            want_scan = True
        except EnvironmentError:
            tcp_params = _winreg.OpenKey(lm, 'SYSTEM\\CurrentControlSet\\Services\\VxD\\MSTCP')
        try:
            self._config_win32_fromkey(tcp_params, True)
        finally:
            tcp_params.Close()
        if want_scan:
            interfaces = _winreg.OpenKey(lm, 'SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters\\Interfaces')
            try:
                i = 0
                while True:
                    try:
                        guid = _winreg.EnumKey(interfaces, i)
                        i += 1
                        key = _winreg.OpenKey(interfaces, guid)
                        if not self._win32_is_nic_enabled(lm, guid, key):
                            continue
                        try:
                            self._config_win32_fromkey(key, False)
                        finally:
                            key.Close()
                    except EnvironmentError:
                        break
            finally:
                interfaces.Close()
    finally:
        lm.Close()