import configparser
import os
from paste.deploy import loadapp
from gunicorn.app.wsgiapp import WSGIApplication
from gunicorn.config import get_default_config_file
def has_logging_config(config_file):
    parser = configparser.ConfigParser()
    parser.read([config_file])
    return parser.has_section('loggers')