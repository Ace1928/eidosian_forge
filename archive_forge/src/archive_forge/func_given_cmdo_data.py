from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def given_cmdo_data(self):
    cmdo = 'image server'
    data = [('image', 'create'), ('image_create', '--eolus'), ('server', 'meta ssh'), ('server_meta_delete', '--wilson'), ('server_ssh', '--sunlight')]
    return (cmdo, data)