from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
@contextmanager
def wait_page_loaded(self, timeout=10):
    """
        Block until a new page has loaded and is ready.
        """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as ec
    old_page = self.selenium.find_element(By.TAG_NAME, 'html')
    yield
    self.wait_until(ec.staleness_of(old_page), timeout=timeout)
    self.wait_page_ready(timeout=timeout)