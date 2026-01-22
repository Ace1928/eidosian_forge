import json
from collections import namedtuple
import macaroonbakery.bakery as bakery
def request_version(req_headers):
    """ Determines the bakery protocol version from a client request.
    If the protocol cannot be determined, or is invalid, the original version
    of the protocol is used. If a later version is found, the latest known
    version is used, which is OK because versions are backwardly compatible.

    @param req_headers: the request headers as a dict.
    @return: bakery protocol version (for example macaroonbakery.VERSION_1)
    """
    vs = req_headers.get(BAKERY_PROTOCOL_HEADER)
    if vs is None:
        return bakery.VERSION_1
    try:
        x = int(vs)
    except ValueError:
        return bakery.VERSION_1
    if x > bakery.LATEST_VERSION:
        return bakery.LATEST_VERSION
    return x