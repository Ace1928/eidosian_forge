import typing
from selenium.webdriver.common import service
@reuse_service.setter
def reuse_service(self, reuse: bool) -> None:
    if not isinstance(reuse, bool):
        raise TypeError('reuse must be a boolean')
    self._reuse_service = reuse