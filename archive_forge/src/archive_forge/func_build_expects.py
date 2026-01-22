from copy import deepcopy
def build_expects(self, fields=None):
    """
        Builds up a list of expecations to hand off to DynamoDB on save.

        Largely internal.
        """
    expects = {}
    if fields is None:
        fields = list(self._data.keys()) + list(self._orig_data.keys())
    fields = set(fields)
    for key in fields:
        expects[key] = {'Exists': True}
        value = None
        if not key in self._orig_data and (not key in self._data):
            raise ValueError('Unknown key %s provided.' % key)
        orig_value = self._orig_data.get(key, NEWVALUE)
        current_value = self._data.get(key, NEWVALUE)
        if orig_value == current_value:
            value = current_value
        elif key in self._data:
            if not key in self._orig_data:
                expects[key]['Exists'] = False
            else:
                value = orig_value
        else:
            value = orig_value
        if value is not None:
            expects[key]['Value'] = self._dynamizer.encode(value)
    return expects