@property
def is_to_be_committed(self):
    """"Determine if the event payload resource is to be committed.

        :returns:  True if the desired state has been populated, else False.
        """
    return self.desired_state is not None