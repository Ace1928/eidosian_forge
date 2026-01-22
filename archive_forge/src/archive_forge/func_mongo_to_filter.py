def mongo_to_filter(self, filter):
    if filter is None:
        return None
    group_op = None
    for key in filter.keys():
        if key in self.MONGO_TO_GROUP_OP:
            group_op = key
            break
    if group_op is not None:
        return {'op': self.MONGO_TO_GROUP_OP[group_op], 'filters': [self.mongo_to_filter(f) for f in filter[group_op]]}
    else:
        for k, v in filter.items():
            if isinstance(v, dict):
                op = next(iter(v.keys()))
                return {'key': self.server_path_to_key(k), 'op': self.MONGO_TO_INDIVIDUAL_OP[op], 'value': v[op]}
            else:
                return {'key': self.server_path_to_key(k), 'op': '=', 'value': v}