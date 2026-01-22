import os
import time
import numpy as np
import PyChromeDevTools
import ipyvolume as ipv
Generate images from ipyvolume using chrome headless.

Assuming osx, define the following aliases for convenience, and start in headless mode::

     $ alias chrome="/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome"
     $ chrome --remote-debugging-port=9222 --headless

Make sure you have `PyChromeDevTools` installed::

    $ pip install PyChromeDevTools

Now run the following snippet (doesn't have to be from the Jupyter notebook) ::

    import ipyvolume as ipv
    ipv.examples.klein_bottle()
    ipv.view(10,30)
    ipv.savefig('headless.png', headless=True)


