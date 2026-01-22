from bs4 import BeautifulSoup
@property
def dash_innerhtml_dom(self):
    return self._get_dash_dom_by_attribute('innerHTML')