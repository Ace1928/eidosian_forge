from bs4 import BeautifulSoup
def get_session_storage(self, session_id='session'):
    return self.driver.execute_script(f"return JSON.parse(window.sessionStorage.getItem('{session_id}'));")