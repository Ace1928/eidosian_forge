Translates the state machine into a pydot graph.

    :param machine: state machine to convert
    :type machine: FiniteMachine
    :param graph_name: name of the graph to be created
    :type graph_name: string
    :param graph_attrs: any initial graph attributes to set
                        (see http://www.graphviz.org/doc/info/attrs.html for
                        what these can be)
    :type graph_attrs: dict
    :param node_attrs_cb: a callback that takes one argument ``state``
                          and is expected to return a dict of node attributes
                          (see http://www.graphviz.org/doc/info/attrs.html for
                          what these can be)
    :type node_attrs_cb: callback
    :param edge_attrs_cb: a callback that takes three arguments ``start_state,
                          event, end_state`` and is expected to return a dict
                          of edge attributes (see
                          http://www.graphviz.org/doc/info/attrs.html for
                          what these can be)
    :type edge_attrs_cb: callback
    :param add_start_state: when enabled this creates a *private* start state
                            with the name ``__start__`` that will be a point
                            node that will have a dotted edge to the
                            ``default_start_state`` that your machine may have
                            defined (if your machine has no actively defined
                            ``default_start_state`` then this does nothing,
                            even if enabled)
    :type add_start_state: bool
    :param name_translations: a dict that provides alternative ``state``
                              string names for each state
    :type name_translations: dict
    