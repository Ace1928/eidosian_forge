def get_nsmap(node):
    nsmap = node.nsmap
    none_ns = node.nsmap.get(None)
    if none_ns is None:
        nsmap[none_ns.rsplit('/', 1)[1]] = none_ns
        del nsmap[None]
    return nsmap