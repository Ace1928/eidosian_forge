def nvl(self, value=None):
    """
        Convert the I{value} into the default when I{None}.
        @param value: The proposed value.
        @type value: any
        @return: The I{default} when I{value} is I{None}, else I{value}.
        @rtype: any
        """
    if value is None:
        return self.default
    else:
        return value