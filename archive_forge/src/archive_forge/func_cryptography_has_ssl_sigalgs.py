from __future__ import annotations
import typing
def cryptography_has_ssl_sigalgs() -> typing.List[str]:
    return ['SSL_CTX_set1_sigalgs_list']