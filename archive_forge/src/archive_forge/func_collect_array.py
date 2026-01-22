def collect_array(a_value, base, nodes):
    a_type = a_value['name']
    if is_node(a_type):
        nodes.append(base)
    elif a_type in ('shape', 'exact'):
        nodes = collect_nodes(a_value['value'], base + '[]', nodes)
    elif a_type == 'union':
        nodes = collect_union(a_value['value'], base + '[]', nodes)
    elif a_type == 'objectOf':
        nodes = collect_object(a_value['value'], base + '[]', nodes)
    return nodes