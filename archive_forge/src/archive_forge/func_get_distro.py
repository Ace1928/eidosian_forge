import os
import pkg_resources
def get_distro(spec):
    return pkg_resources.get_distribution(spec)