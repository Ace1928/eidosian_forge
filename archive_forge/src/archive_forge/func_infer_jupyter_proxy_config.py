import asyncio
import io
import inspect
import logging
import os
import queue
import uuid
import sys
import threading
import time
from typing_extensions import Literal
from werkzeug.serving import make_server
def infer_jupyter_proxy_config(self):
    """
        Infer the current Jupyter server configuration. This will detect
        the proper request_pathname_prefix and server_url values to use when
        displaying Dash apps.Dash requests will be routed through the proxy.

        Requirements:

        In the classic notebook, this method requires the `dash` nbextension
        which should be installed automatically with the installation of the
        jupyter-dash Python package. You can see what notebook extensions are installed
        by running the following command:
            $ jupyter nbextension list

        In JupyterLab, this method requires the `@plotly/dash-jupyterlab` labextension. This
        extension should be installed automatically with the installation of the
        jupyter-dash Python package, but JupyterLab must be allowed to rebuild before
        the extension is activated (JupyterLab should automatically detect the
        extension and produce a popup dialog asking for permission to rebuild). You can
        see what JupyterLab extensions are installed by running the following command:
            $ jupyter labextension list
        """
    if not self.in_ipython or self.in_colab:
        return
    _request_jupyter_config()