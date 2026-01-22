from defusedxml.lxml import parse
def get_vmdk_name_from_ovf(ovf_handle):
    """Get the vmdk name from the given ovf descriptor."""
    return _get_vmdk_name_from_ovf(parse(ovf_handle).getroot())