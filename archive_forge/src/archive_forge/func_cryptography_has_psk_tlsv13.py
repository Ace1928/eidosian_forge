from __future__ import annotations
import typing
def cryptography_has_psk_tlsv13() -> typing.List[str]:
    return ['SSL_CTX_set_psk_find_session_callback', 'SSL_CTX_set_psk_use_session_callback', 'Cryptography_SSL_SESSION_new', 'SSL_CIPHER_find', 'SSL_SESSION_set1_master_key', 'SSL_SESSION_set_cipher', 'SSL_SESSION_set_protocol_version']