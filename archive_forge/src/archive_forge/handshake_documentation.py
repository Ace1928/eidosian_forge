from consonance.protocol import WANoiseProtocol
from consonance.streams.segmented.segmented import SegmentedStream
from consonance.exceptions.handshake_failed_exception import HandshakeFailedException
from consonance.config.client import ClientConfig
from consonance.structs.keypair import KeyPair
from consonance.structs.publickey import PublicKey
import threading
import logging

        :param wanoiseprotocol:
        :type wanoiseprotocol: WANoiseProtocol
        :param stream:
        :type stream: SegmentedStream
        :param client_config:
        :type client_config: ClientConfig
        :param s:
        :type s: KeyPair
        :param rs:
        :type rs: PublicKey | None
        