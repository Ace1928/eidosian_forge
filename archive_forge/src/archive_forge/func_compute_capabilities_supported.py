def compute_capabilities_supported(sdk_trait):
    """Given an SDK trait, returns a set of compute capability traits that the
    version of the SDK supports.

    Returns None if no matches were found for the SDK trait.
    """
    return _SDK_COMPUTE_CAP_SUPPORT.get(sdk_trait)