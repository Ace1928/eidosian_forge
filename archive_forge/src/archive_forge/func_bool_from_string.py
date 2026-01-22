import warnings
def bool_from_string(s, accepted_values=None):
    """Returns a boolean if the string can be interpreted as such.

    Interpret case insensitive strings as booleans. The default values
    includes: 'yes', 'no, 'y', 'n', 'true', 'false', '0', '1', 'on',
    'off'. Alternative values can be provided with the 'accepted_values'
    parameter.

    Args:
      s: A string that should be interpreted as a boolean. It should be of
         type string or unicode.

      accepted_values: An optional dict with accepted strings as keys and
         True/False as values. The strings will be tested against a lowered
         version of 's'.

    Returns:
      True or False for accepted strings, None otherwise.
    """
    if accepted_values is None:
        accepted_values = _valid_boolean_strings
    val = None
    if isinstance(s, str):
        try:
            val = accepted_values[s.lower()]
        except KeyError:
            pass
    return val