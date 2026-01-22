from octavia_lib.api.drivers import exceptions
def l7rule_delete(self, l7rule):
    """Deletes an L7 rule.

        :param l7rule: The L7 rule to delete.
        :type l7rule: object
        :return: Nothing if the delete request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support deleting l7rules.', operator_fault_string='This provider does not support deleting l7rules.')