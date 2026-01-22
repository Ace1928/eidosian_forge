import time
def timings(reps, func, *args, **kw):
    """timings(reps,func,*args,**kw) -> (t_total,t_per_call)

    Execute a function reps times, return a tuple with the elapsed total CPU
    time in seconds and the time per call. These are just the first two values
    in timings_out()."""
    return timings_out(reps, func, *args, **kw)[0:2]