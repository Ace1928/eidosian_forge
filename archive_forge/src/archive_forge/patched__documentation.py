import os
import pkgutil
import shutil
import tempfile
import httplib2
Patch things so that httplib2 works properly in a PAR.

  Manually extract certificates to file to make OpenSSL happy and avoid error:
     ssl.SSLError: [Errno 185090050] _ssl.c:344: error:0B084002:x509 ...

  Args:
    extract_dir: the directory into which we extract the necessary files.
  