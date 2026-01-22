def edns_to_text(flags):
    """Convert an EDNS flags value into a space-separated list of EDNS flag
    text values.

    Returns a ``text``.
    """
    return _to_text(flags, _edns_by_value, _edns_flags_order)