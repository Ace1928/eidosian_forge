import atexit
import os
import signal
import sys
import time

This is a lightweight "reaper" process used to ensure that ray processes are
cleaned up properly when the main ray process dies unexpectedly (e.g.,
segfaults or gets SIGKILLed). Note that processes may not be cleaned up
properly if this process is SIGTERMed or SIGKILLed.

It detects that its parent has died by reading from stdin, which must be
inherited from the parent process so that the OS will deliver an EOF if the
parent dies. When this happens, the reaper process kills the rest of its
process group (first attempting graceful shutdown with SIGTERM, then escalating
to SIGKILL).
