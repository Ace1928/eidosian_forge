@property
def has_states(self):
    """Determines if this event payload has any states.

        :returns: True if this event payload has states, otherwise False.
        """
    return len(self.states) > 0