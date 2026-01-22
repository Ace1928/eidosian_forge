import os
import sys
from ...interfaces.base import CommandLine
from .base import GraphPluginBase, logger
def make_job_name(jobnumber, nodeslist):
    """
            - jobnumber: The index number of the job to create
            - nodeslist: The name of the node being processed
            - return: A string representing this job to be displayed by SLURM
            """
    job_name = 'j{0}_{1}'.format(jobnumber, nodeslist[jobnumber]._id)
    job_name = job_name.replace('-', '_').replace('.', '_').replace(':', '_')
    return job_name