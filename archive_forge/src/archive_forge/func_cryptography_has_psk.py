from __future__ import annotations
import typing
def cryptography_has_psk() -> typing.List[str]:
    return ['SSL_CTX_use_psk_identity_hint', 'SSL_CTX_set_psk_server_callback', 'SSL_CTX_set_psk_client_callback']