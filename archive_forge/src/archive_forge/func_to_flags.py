import dns.exception
def to_flags(value):
    """Convert an opcode to a value suitable for ORing into DNS message
    flags.

    *value*, an ``int``, the DNS opcode value.

    Returns an ``int``.
    """
    return value << 11 & 30720