from octavia_lib.api.drivers import exceptions
def l7policy_update(self, old_l7policy, new_l7policy):
    """Updates an L7 policy.

        :param old_l7policy: The baseline L7 policy object.
        :type old_l7policy: object
        :param new_l7policy: The updated L7 policy object.
        :type new_l7policy: object
        :return: Nothing if the update request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support updating l7policies.', operator_fault_string='This provider does not support updating l7policies.')