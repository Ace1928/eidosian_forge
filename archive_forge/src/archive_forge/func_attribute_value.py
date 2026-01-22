from bs4.dammit import EntitySubstitution
def attribute_value(self, value):
    """Process the value of an attribute.

        :param ns: A string.
        :return: A string with certain characters replaced by named
           or numeric entities.
        """
    return self.substitute(value)