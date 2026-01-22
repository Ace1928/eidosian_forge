def order_to_keys(self, order):
    keys = []
    for k in order:
        name = k[1:]
        if k[0] == '+':
            ascending = True
        elif k[0] == '-':
            ascending = False
        else:
            raise Exception('you must sort by ascending(+) or descending(-)')
        key = {'key': {'section': 'run', 'name': name}, 'ascending': ascending}
        keys.append(key)
    return {'keys': keys}