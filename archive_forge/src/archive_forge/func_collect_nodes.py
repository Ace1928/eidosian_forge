def collect_nodes(metadata, base='', nodes=None):
    nodes = nodes or []
    for prop_name, value in metadata.items():
        t_value = value.get('type', value)
        p_type = t_value.get('name')
        if base:
            key = f'{base}.{prop_name}'
        else:
            key = prop_name
        if is_node(p_type):
            nodes.append(key)
        elif p_type == 'arrayOf':
            a_value = t_value.get('value', t_value)
            nodes = collect_array(a_value, key, nodes)
        elif is_shape(p_type):
            nodes = collect_nodes(t_value['value'], key, nodes)
        elif p_type == 'union':
            nodes = collect_union(t_value['value'], key, nodes)
        elif p_type == 'objectOf':
            o_value = t_value.get('value', {})
            nodes = collect_object(o_value, key, nodes)
    return nodes