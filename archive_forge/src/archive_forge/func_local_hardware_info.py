import multiprocessing as mp
import platform
import os
def local_hardware_info():
    """Basic hardware information about the local machine.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on. CPU count defaults to 1 when true count can't be determined.

    Returns:
        dict: The hardware information.
    """
    if hasattr(os, 'sched_getaffinity'):
        num_cpus = len(os.sched_getaffinity(0))
    else:
        num_cpus = os.cpu_count()
    if num_cpus is None:
        num_cpus = 1
    else:
        num_cpus = int(num_cpus / 2) or 1
    results = {'python_compiler': platform.python_compiler(), 'python_build': ', '.join(platform.python_build()), 'python_version': platform.python_version(), 'os': platform.system(), 'cpus': num_cpus}
    return results