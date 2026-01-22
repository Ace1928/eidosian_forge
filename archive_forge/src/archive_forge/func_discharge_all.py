import abc
from collections import namedtuple
from ._error import (
from ._codec import decode_caveat
from ._macaroon import (
from ._versions import VERSION_2
from ._third_party import ThirdPartyCaveatInfo
import macaroonbakery.checkers as checkers
def discharge_all(m, get_discharge, local_key=None):
    """Gathers discharge macaroons for all the third party caveats in m
    (and any subsequent caveats required by those) using get_discharge to
    acquire each discharge macaroon.
    The local_key parameter may optionally hold the key of the client, in
    which case it will be used to discharge any third party caveats with the
    special location "local". In this case, the caveat itself must be "true".
    This can be used be a server to ask a client to prove ownership of the
    private key.
    It returns a list of macaroon with m as the first element, followed by all
    the discharge macaroons.
    All the discharge macaroons will be bound to the primary macaroon.

    The get_discharge function is passed a context (AuthContext),
    the caveat(pymacaroons.Caveat) to be discharged and encrypted_caveat (bytes) will be
    passed the external caveat payload found in m, if any.
    It should return a bakery.Macaroon object holding the discharge
    macaroon for the third party caveat.
    """
    primary = m.macaroon
    discharges = [primary]
    _NeedCaveat = namedtuple('_NeedCaveat', 'cav encrypted_caveat')
    need = []

    def add_caveats(m):
        for cav in m.macaroon.caveats:
            if cav.location is None or cav.location == '':
                continue
            encrypted_caveat = m.caveat_data.get(cav.caveat_id, None)
            need.append(_NeedCaveat(cav=cav, encrypted_caveat=encrypted_caveat))
    add_caveats(m)
    while len(need) > 0:
        cav = need[0]
        need = need[1:]
        if cav.cav.location == 'local':
            if local_key is None:
                raise ThirdPartyCaveatCheckFailed('found local third party caveat but no private key provided')
            dm = discharge(ctx=emptyContext, key=local_key, checker=_LocalDischargeChecker(), caveat=cav.encrypted_caveat, id=cav.cav.caveat_id_bytes, locator=_EmptyLocator())
        else:
            dm = get_discharge(cav.cav, cav.encrypted_caveat)
        discharge_m = dm.macaroon
        m = primary.prepare_for_request(discharge_m)
        discharges.append(m)
        add_caveats(dm)
    return discharges