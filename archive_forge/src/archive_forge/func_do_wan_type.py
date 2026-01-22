import sys, re, curl, exceptions
from the command line first, then standard input.
def do_wan_type(self, line):
    try:
        type = eval('LinksysSession.WAN_CONNECT_' + line.strip().upper())
        self.session.set_connection_type(type)
    except ValueError:
        print_stderr('linksys: unknown connection type.')
    return 0