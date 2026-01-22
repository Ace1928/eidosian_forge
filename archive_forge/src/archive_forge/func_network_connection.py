from .command import Command
@property
def network_connection(self):
    return self.ConnectionType(self._driver.execute(Command.GET_NETWORK_CONNECTION)['value'])