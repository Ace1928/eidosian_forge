from octavia_lib.api.drivers import exceptions
def validate_availability_zone(self, availability_zone_metadata):
    """Validates if driver can support the availability zone.

        :param availability_zone_metadata: Dictionary with az metadata.
        :type availability_zone_metadata: dict
        :return: Nothing if the availability zone is valid and supported.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: The driver does not support availability
          zones.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support validating availability zones.', operator_fault_string='This provider does not support validating the supported availability zone metadata.')