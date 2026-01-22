from __future__ import print_function
import os
import io
import time
import functools
import collections
import collections.abc
import numpy as np
import requests
import IPython
def colab_workarounds():
    if environment() == 'colab':
        import IPython.display
        global _colab_enabled_custom_widget_manager
        if not _colab_enabled_custom_widget_manager:
            from google.colab import output
            output.enable_custom_widget_manager()
            _colab_enabled_custom_widget_manager = True
        import ipyvue
        IPython.display.display(ipyvue.Html(tag='span', style_='display: none'))