import time
def clocks():
    """clocks() -> floating point number

        Return the *SYSTEM* CPU time in seconds since the start of the process.
        This is done via a call to resource.getrusage, so it avoids the
        wraparound problems in time.clock()."""
    return resource.getrusage(resource.RUSAGE_SELF)[1]