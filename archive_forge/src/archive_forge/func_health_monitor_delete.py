from octavia_lib.api.drivers import exceptions
def health_monitor_delete(self, healthmonitor):
    """Deletes a healthmonitor_id.

        :param healthmonitor: The monitor to delete.
        :type healthmonitor: object
        :return: Nothing if the create request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: if driver does not support request.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support deleting health monitors.', operator_fault_string='This provider does not support deleting health monitors.')