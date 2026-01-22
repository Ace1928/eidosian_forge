from __future__ import annotations
import typing
def cryptography_has_bn_flags() -> typing.List[str]:
    return ['BN_FLG_CONSTTIME', 'BN_set_flags', 'BN_prime_checks_for_size']