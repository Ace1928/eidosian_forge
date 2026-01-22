import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
Stops the Hyper-V jobs associated with the given resource.

        :param element: string representing the path of the Hyper-V resource
            whose jobs will be stopped.
        :param timeout: the maximum amount of time allowed to stop all the
            given resource's jobs.
        :raises exceptions.JobTerminateFailed: if there are still pending jobs
            associated with the given resource and the given timeout amount of
            time has passed.
        