import sys
import unittest
from contextlib import contextmanager
from django.test import LiveServerTestCase, tag
from django.utils.functional import classproperty
from django.utils.module_loading import import_string
from django.utils.text import capfirst
def create_webdriver(self):
    options = self.create_options()
    if self.selenium_hub:
        from selenium import webdriver
        for key, value in self.get_capability(self.browser).items():
            options.set_capability(key, value)
        return webdriver.Remote(command_executor=self.selenium_hub, options=options)
    return self.import_webdriver(self.browser)(options=options)