from octavia_lib.api.drivers import exceptions
def l7rule_update(self, old_l7rule, new_l7rule):
    """Updates an L7 rule.

        :param old_l7rule: The baseline L7 rule object.
        :type old_l7rule: object
        :param new_l7rule: The updated L7 rule object.
        :type new_l7rule: object
        :return: Nothing if the update request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support updating l7rules.', operator_fault_string='This provider does not support updating l7rules.')