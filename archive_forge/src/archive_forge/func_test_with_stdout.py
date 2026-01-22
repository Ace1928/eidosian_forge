import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
def test_with_stdout(self):
    stdout = "\n        Lo, praise of the prowess of people-kings\n        of spear-armed Danes, in days long sped,\n        we have heard, and what honor the athelings won!\n        Oft Scyld the Scefing from squadroned foes,\n        from many a tribe, the mead-bench tore,\n        awing the earls. Since erst he lay\n        friendless, a foundling, fate repaid him:\n        for he waxed under welkin, in wealth he throve,\n        till before him the folk, both far and near,\n        who house by the whale-path, heard his mandate,\n        gave him gifts: a good king he!\n        To him an heir was afterward born,\n        a son in his halls, whom heaven sent\n        to favor the folk, feeling their woe\n        that erst they had lacked an earl for leader\n        so long a while; the Lord endowed him,\n        the Wielder of Wonder, with world's renown.\n        ".strip()
    err = processutils.ProcessExecutionError(stdout=stdout)
    print(str(err))
    self.assertIn('people-kings', str(err))