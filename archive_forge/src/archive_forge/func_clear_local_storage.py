from bs4 import BeautifulSoup
def clear_local_storage(self):
    self.driver.execute_script('window.localStorage.clear()')