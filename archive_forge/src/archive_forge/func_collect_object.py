def collect_object(o_value, base, nodes):
    o_name = o_value.get('name')
    o_key = base + '{}'
    if is_node(o_name):
        nodes.append(o_key)
    elif is_shape(o_name):
        nodes = collect_nodes(o_value.get('value', {}), o_key, nodes)
    elif o_name == 'union':
        nodes = collect_union(o_value.get('value'), o_key, nodes)
    elif o_name == 'arrayOf':
        nodes = collect_array(o_value, o_key, nodes)
    return nodes