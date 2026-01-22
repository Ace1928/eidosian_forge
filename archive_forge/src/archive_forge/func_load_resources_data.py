def load_resources_data(data):
    """Return the data for all of the resources that meet at a SyncPoint.

    The input is the input_data dict from a SyncPoint received over RPC. The
    keys (which are ignored) are resource primary keys.

    The output is a dict of NodeData objects with the resource names as the
    keys.
    """
    nodes = (NodeData.from_dict(nd) for nd in data.values() if nd is not None)
    return {node.name: node for node in nodes}