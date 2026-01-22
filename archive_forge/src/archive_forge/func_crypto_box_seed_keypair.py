from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def crypto_box_seed_keypair(seed: bytes) -> Tuple[bytes, bytes]:
    """
    Returns a (public, secret) keypair deterministically generated
    from an input ``seed``.

    .. warning:: The seed **must** be high-entropy; therefore,
        its generator **must** be a cryptographic quality
        random function like, for example, :func:`~nacl.utils.random`.

    .. warning:: The seed **must** be protected and remain secret.
        Anyone who knows the seed is really in possession of
        the corresponding PrivateKey.


    :param seed: bytes
    :rtype: (bytes(public_key), bytes(secret_key))
    """
    ensure(isinstance(seed, bytes), 'seed must be bytes', raising=TypeError)
    if len(seed) != crypto_box_SEEDBYTES:
        raise exc.ValueError('Invalid seed')
    pk = ffi.new('unsigned char[]', crypto_box_PUBLICKEYBYTES)
    sk = ffi.new('unsigned char[]', crypto_box_SECRETKEYBYTES)
    rc = lib.crypto_box_seed_keypair(pk, sk, seed)
    ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)
    return (ffi.buffer(pk, crypto_box_PUBLICKEYBYTES)[:], ffi.buffer(sk, crypto_box_SECRETKEYBYTES)[:])