import collections
def decimal_delimiter(self, value):
    self._validate_char(value)
    self._locale['decimal'] = value
    return self