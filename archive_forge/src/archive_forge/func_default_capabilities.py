import typing
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
@property
def default_capabilities(self) -> typing.Dict[str, str]:
    return DesiredCapabilities.WPEWEBKIT.copy()