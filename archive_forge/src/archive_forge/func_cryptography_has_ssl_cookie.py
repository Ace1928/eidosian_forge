from __future__ import annotations
import typing
def cryptography_has_ssl_cookie() -> typing.List[str]:
    return ['SSL_OP_COOKIE_EXCHANGE', 'DTLSv1_listen', 'SSL_CTX_set_cookie_generate_cb', 'SSL_CTX_set_cookie_verify_cb']