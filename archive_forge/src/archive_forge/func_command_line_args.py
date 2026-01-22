import shutil
import typing
from selenium.webdriver.common import service
def command_line_args(self) -> typing.List[str]:
    return ['-p', f'{self.port}'] + self.service_args