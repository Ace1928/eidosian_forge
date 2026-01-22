from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_kx_server_session_keys(server_public_key: bytes, server_secret_key: bytes, client_public_key: bytes) -> Tuple[bytes, bytes]:
    """
    Generate session keys for the server.
    :param server_public_key:
    :type server_public_key: bytes
    :param server_secret_key:
    :type server_secret_key: bytes
    :param client_public_key:
    :type client_public_key: bytes
    :return: (rx_key, tx_key)
    :rtype: (bytes, bytes)
    """
    ensure(isinstance(server_public_key, bytes) and len(server_public_key) == crypto_kx_PUBLIC_KEY_BYTES, 'Server public key must be a {} bytes long bytes sequence'.format(crypto_kx_PUBLIC_KEY_BYTES), raising=exc.TypeError)
    ensure(isinstance(server_secret_key, bytes) and len(server_secret_key) == crypto_kx_SECRET_KEY_BYTES, 'Server secret key must be a {} bytes long bytes sequence'.format(crypto_kx_PUBLIC_KEY_BYTES), raising=exc.TypeError)
    ensure(isinstance(client_public_key, bytes) and len(client_public_key) == crypto_kx_PUBLIC_KEY_BYTES, 'Client public key must be a {} bytes long bytes sequence'.format(crypto_kx_PUBLIC_KEY_BYTES), raising=exc.TypeError)
    rx_key = ffi.new('unsigned char[]', crypto_kx_SESSION_KEY_BYTES)
    tx_key = ffi.new('unsigned char[]', crypto_kx_SESSION_KEY_BYTES)
    res = lib.crypto_kx_server_session_keys(rx_key, tx_key, server_public_key, server_secret_key, client_public_key)
    ensure(res == 0, 'Server session key generation failed.', raising=exc.CryptoError)
    return (ffi.buffer(rx_key, crypto_kx_SESSION_KEY_BYTES)[:], ffi.buffer(tx_key, crypto_kx_SESSION_KEY_BYTES)[:])