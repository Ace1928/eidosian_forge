import sys, re, curl, exceptions
from the command line first, then standard input.
def set_IP_address(self, page, cgi, role, ip):
    ind = 0
    for octet in ip.split('.'):
        self.actions.append(('', 'F1', role + repr(ind + 1), octet))
        ind += 1