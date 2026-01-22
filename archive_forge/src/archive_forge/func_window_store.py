from bs4 import BeautifulSoup
@property
def window_store(self):
    return self.driver.execute_script('return window.store')