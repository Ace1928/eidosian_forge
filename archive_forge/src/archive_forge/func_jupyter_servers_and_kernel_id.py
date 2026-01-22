import json
import logging
import os
import re
import shutil
import sys
from base64 import b64encode
from typing import Dict
import requests
from requests.compat import urljoin
import wandb
import wandb.util
from wandb.sdk.lib import filesystem
def jupyter_servers_and_kernel_id():
    """Return a list of servers and the current kernel_id.

    Used to query for the name of the notebook.
    """
    try:
        import ipykernel
        kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)
        serverapp = wandb.util.get_module('jupyter_server.serverapp')
        notebookapp = wandb.util.get_module('notebook.notebookapp')
        servers = []
        if serverapp is not None:
            servers.extend(list(serverapp.list_running_servers()))
        if notebookapp is not None:
            servers.extend(list(notebookapp.list_running_servers()))
        return (servers, kernel_id)
    except (AttributeError, ValueError, ImportError):
        return ([], None)