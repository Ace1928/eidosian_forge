def prepare_graph_for_ui(graph, limit_attr_size=1024, large_attrs_key='_too_large_attrs'):
    """Prepares (modifies in-place) the graph to be served to the front-end.

    For now, it supports filtering out attributes that are
    too large to be shown in the graph UI.

    Args:
      graph: The GraphDef proto message.
      limit_attr_size: Maximum allowed size in bytes, before the attribute
          is considered large. Default is 1024 (1KB). Must be > 0 or None.
          If None, there will be no filtering.
      large_attrs_key: The attribute key that will be used for storing attributes
          that are too large. Default is '_too_large_attrs'. Must be != None if
          `limit_attr_size` is != None.

    Raises:
      ValueError: If `large_attrs_key is None` while `limit_attr_size != None`.
      ValueError: If `limit_attr_size` is defined, but <= 0.
    """
    if limit_attr_size is not None:
        if large_attrs_key is None:
            raise ValueError('large_attrs_key must be != None when limit_attr_size!= None.')
        if limit_attr_size <= 0:
            raise ValueError('limit_attr_size must be > 0, but is %d' % limit_attr_size)
    if limit_attr_size is not None:
        for node in graph.node:
            keys = list(node.attr.keys())
            for key in keys:
                size = node.attr[key].ByteSize()
                if size > limit_attr_size or size < 0:
                    del node.attr[key]
                    node.attr[large_attrs_key].list.s.append(key.encode('utf-8'))