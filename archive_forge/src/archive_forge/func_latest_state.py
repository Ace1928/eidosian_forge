@property
def latest_state(self):
    """Returns the latest state for the event payload resource.

        :returns: If this payload has a desired_state its returned, otherwise
            latest_state is returned.
        """
    return self.desired_state or super().latest_state