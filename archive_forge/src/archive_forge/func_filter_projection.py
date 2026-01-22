def filter_projection(left, right, comparator):
    return {'type': 'filter_projection', 'children': [left, right, comparator]}