import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def qry(self, *name, **args):
    """
        Request function for the DnsRequest class.  In addition to standard
        DNS args, the special pydns arg 'resulttype' can optionally be passed.
        Valid resulttypes are 'default', 'text', 'decimal', and 'binary'.

        Defaults are configured to be compatible with pydns:
        AAAA: decimal
        Others: text
        """
    ' needs a refactoring '
    self.argparse(name, args)
    protocol = self.args['protocol']
    self.port = self.args['port']
    self.tid = random.randint(0, 65535)
    self.timeout = self.args['timeout']
    opcode = self.args['opcode']
    rd = self.args['rd']
    server = self.args['server']
    if 'resulttype' in self.args:
        self.resulttype = self.args['resulttype']
    else:
        self.resulttype = 'default'
    if type(self.args['qtype']) == bytes or type(self.args['qtype']) == str:
        try:
            qtype = getattr(Type, str(self.args['qtype'].upper()))
        except AttributeError:
            raise ArgumentError('unknown query type')
    else:
        qtype = self.args['qtype']
    if 'name' not in self.args:
        print(self.args)
        raise ArgumentError('nothing to lookup')
    qname = self.args['name']
    if qtype == Type.AXFR and protocol != 'tcp':
        print('Query type AXFR, protocol forced to TCP')
        protocol = 'tcp'
    m = Lib.Mpacker()
    m.addHeader(self.tid, 0, opcode, 0, 0, rd, 0, 0, 0, 1, 0, 0, 0)
    m.addQuestion(qname, qtype, Class.IN)
    self.request = m.getbuf()
    try:
        if protocol == 'udp':
            self.sendUDPRequest(server)
        else:
            self.sendTCPRequest(server)
    except socket.error as reason:
        raise SocketError(reason)
    return self.response