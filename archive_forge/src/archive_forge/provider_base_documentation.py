from octavia_lib.api.drivers import exceptions
Validates if driver can support the availability zone.

        :param availability_zone_metadata: Dictionary with az metadata.
        :type availability_zone_metadata: dict
        :return: Nothing if the availability zone is valid and supported.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: The driver does not support availability
          zones.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        