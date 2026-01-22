from pprint import pformat
from six import iteritems
import re
@job_template.setter
def job_template(self, job_template):
    """
        Sets the job_template of this V2alpha1CronJobSpec.
        Specifies the job that will be created when executing a CronJob.

        :param job_template: The job_template of this V2alpha1CronJobSpec.
        :type: V2alpha1JobTemplateSpec
        """
    if job_template is None:
        raise ValueError('Invalid value for `job_template`, must not be `None`')
    self._job_template = job_template