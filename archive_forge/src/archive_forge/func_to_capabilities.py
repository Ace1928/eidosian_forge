import typing
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
def to_capabilities(self):
    """Creates a capabilities with all the options that have been set and
        returns a dictionary with everything."""
    caps = self._caps
    browser_options = {}
    if self.binary_location:
        browser_options['binary'] = self.binary_location
    if self.arguments:
        browser_options['args'] = self.arguments
    caps[Options.KEY] = browser_options
    return caps