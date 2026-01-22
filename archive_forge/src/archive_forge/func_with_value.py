def with_value(self, key, val):
    """ Return a copy of the AuthContext object with the given key and
        value added.
        """
    new_dict = dict(self._dict)
    new_dict[key] = val
    return AuthContext(new_dict)