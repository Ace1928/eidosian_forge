import random
from dissononce.processing.impl.handshakestate import HandshakeState
from dissononce.extras.processing.handshakestate_guarded import GuardedHandshakeState
from dissononce.extras.processing.handshakestate_switchable import SwitchableHandshakeState
from dissononce.processing.handshakepatterns.handshakepattern import HandshakePattern
from dissononce.processing.handshakepatterns.interactive.IK import IKHandshakePattern
from dissononce.processing.handshakepatterns.interactive.XX import XXHandshakePattern
from dissononce.processing.modifiers.fallback import FallbackPatternModifier
from dissononce.processing.impl.cipherstate import CipherState
from dissononce.cipher.aesgcm import AESGCMCipher
from dissononce.hash.sha256 import SHA256Hash
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from dissononce.dh.private import PrivateKey
from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.extras.dh.dangerous.dh_nogen import NoGenDH
from dissononce.exceptions.decrypt import DecryptFailedException
from google.protobuf.message import DecodeError
from .dissononce_extras.processing.symmetricstate_wa import WASymmetricState
from .proto import wa20_pb2
from .streams.segmented.segmented import SegmentedStream
from .certman.certman import CertMan
from .exceptions.new_rs_exception import NewRemoteStaticException
from .config.client import ClientConfig
from .structs.publickey import PublicKey
from .util.byte import ByteUtil
from.exceptions.handshake_failed_exception import HandshakeFailedException
import logging
def perform(self, client_config, stream, s, rs=None, e=None):
    """
        :param client_config:
        :type client_config:
        :param stream:
        :type stream:
        :param s:
        :type s: consonance.structs.keypair.KeyPair
        :param rs:
        :type rs: consonance.structs.publickey.PublicKey | None
        :type e: consonance.structs.keypair.KeyPair | None
        :return:
        :rtype:
        """
    logger.debug('perform(client_config=%s, stream=%s, s=%s, rs=%s, e=%s)' % (client_config, stream, s, rs, e))
    dh = X25519DH()
    if e is not None:
        dh = NoGenDH(dh, PrivateKey(e.private.data))
    self._handshakestate = SwitchableHandshakeState(GuardedHandshakeState(HandshakeState(WASymmetricState(CipherState(AESGCMCipher()), SHA256Hash()), dh)))
    dissononce_s = KeyPair(PublicKey(s.public.data), PrivateKey(s.private.data))
    dissononce_rs = PublicKey(rs.data) if rs else None
    client_payload = self._create_full_payload(client_config)
    logger.debug('Create client_payload=%s' % client_payload)
    try:
        if rs is not None:
            try:
                cipherstatepair = self._start_handshake_ik(stream, client_payload, dissononce_s, dissononce_rs)
            except NewRemoteStaticException as ex:
                cipherstatepair = self._switch_handshake_xxfallback(stream, dissononce_s, client_payload, ex.server_hello)
        else:
            cipherstatepair = self._start_handshake_xx(stream, client_payload, dissononce_s)
        return cipherstatepair
    except DecryptFailedException as e:
        logger.exception(e)
        raise HandshakeFailedException(e)
    except DecodeError as e:
        logger.exception(e)
        raise HandshakeFailedException(e)