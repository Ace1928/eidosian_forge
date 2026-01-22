import pytest
from jupyterlab import __version__
from jupyterlab.handlers.announcements import (
@pytest.fixture
def labserverapp(jp_serverapp, make_labserver_extension_app):
    app = make_labserver_extension_app()
    app._link_jupyter_server_extension(jp_serverapp)
    app.handlers.extend([('/custom/(.*)(?<!\\.js)$', jp_serverapp.web_app.settings['static_handler_class'], {'path': jp_serverapp.web_app.settings['static_custom_path'], 'no_cache_paths': ['/']}), (check_update_handler_path, CheckForUpdateHandler, {'update_checker': CheckForUpdate(__version__)}), (news_handler_path, NewsHandler, {'news_url': 'https://dummy.io/feed.xml'})])
    app.initialize()
    return app