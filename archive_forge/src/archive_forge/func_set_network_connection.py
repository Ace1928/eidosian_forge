from .command import Command
def set_network_connection(self, network):
    """Set the network connection for the remote device.

        Example of setting airplane mode::

            driver.mobile.set_network_connection(driver.mobile.AIRPLANE_MODE)
        """
    mode = network.mask if isinstance(network, self.ConnectionType) else network
    return self.ConnectionType(self._driver.execute(Command.SET_NETWORK_CONNECTION, {'name': 'network_connection', 'parameters': {'type': mode}})['value'])