from . import Base
from . Base import ServerError
def dnslookup(name, qtype, timeout=30):
    """convenience routine to return just answer data for any query type"""
    if Base.defaults['server'] == []:
        Base.DiscoverNameServers()
    result = Base.DnsRequest(name=name, qtype=qtype).req(timeout=timeout)
    if result.header['status'] != 'NOERROR':
        raise ServerError('DNS query status: %s' % result.header['status'], result.header['rcode'])
    elif len(result.answers) == 0 and Base.defaults['server_rotate']:
        result = Base.DnsRequest(name=name, qtype=qtype).req(timeout=timeout)
    if result.header['status'] != 'NOERROR':
        raise ServerError('DNS query status: %s' % result.header['status'], result.header['rcode'])
    return [x['data'] for x in result.answers]