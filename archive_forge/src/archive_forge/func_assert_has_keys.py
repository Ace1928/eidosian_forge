def assert_has_keys(dict, required=None, optional=None):
    required = required or []
    optional = optional or []
    for k in required:
        try:
            assert k in dict
        except AssertionError:
            extra_keys = set(dict).difference(set(required + optional))
            raise AssertionError('found unexpected keys: %s' % list(extra_keys))