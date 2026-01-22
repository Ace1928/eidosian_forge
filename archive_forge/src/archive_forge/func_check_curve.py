from Cryptodome.Util.number import long_to_bytes
from Cryptodome.PublicKey.ECC import EccKey
def check_curve(curve, key, name, private):
    if not isinstance(key, EccKey):
        raise TypeError("'%s' must be an ECC key" % name)
    if private and (not key.has_private()):
        raise TypeError("'%s' must be a private ECC key" % name)
    if curve is None:
        curve = key.curve
    elif curve != key.curve:
        raise TypeError("'%s' is defined on an incompatible curve" % name)
    return curve