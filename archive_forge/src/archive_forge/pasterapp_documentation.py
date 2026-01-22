import configparser
import os
from paste.deploy import loadapp
from gunicorn.app.wsgiapp import WSGIApplication
from gunicorn.config import get_default_config_file
    A Paste Deployment server runner.

    Example configuration:

        [server:main]
        use = egg:gunicorn#main
        host = 127.0.0.1
        port = 5000
    