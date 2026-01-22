import contextlib
import io
import os
import sys
import warnings
Get the username from the environment or password database.

    First try various environment variables, then the password
    database.  This works on Windows as long as USERNAME is set.

    