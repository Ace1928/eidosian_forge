from typing import Optional
import requests
An exception for attempting to get info about an unsuccessful job.

    This exception occurs when a job has been canceled, deleted, or failed, and information about
    this job is attempted to be accessed.
    