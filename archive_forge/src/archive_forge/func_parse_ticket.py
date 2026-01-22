import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
def parse_ticket(secret, ticket, ip, digest_algo=DEFAULT_DIGEST):
    """
    Parse the ticket, returning (timestamp, userid, tokens, user_data).

    If the ticket cannot be parsed, ``BadTicket`` will be raised with
    an explanation.
    """
    if isinstance(digest_algo, bytes):
        digest_algo = getattr(hashlib, digest_algo)
    digest_hexa_size = digest_algo().digest_size * 2
    ticket = ticket.strip(b'"')
    digest = ticket[:digest_hexa_size]
    try:
        timestamp = int(ticket[digest_hexa_size:digest_hexa_size + 8], 16)
    except ValueError as e:
        raise BadTicket('Timestamp is not a hex integer: %s' % e)
    try:
        userid, data = ticket[digest_hexa_size + 8:].split(b'!', 1)
    except ValueError:
        raise BadTicket('userid is not followed by !')
    userid = url_unquote(userid.decode())
    if b'!' in data:
        tokens, user_data = data.split(b'!', 1)
    else:
        tokens = b''
        user_data = data
    expected = calculate_digest(ip, timestamp, secret, userid, tokens, user_data, digest_algo)
    if expected != digest:
        raise BadTicket('Digest signature is not correct', expected=(expected, digest))
    tokens = tokens.split(b',')
    return (timestamp, userid, tokens, user_data)