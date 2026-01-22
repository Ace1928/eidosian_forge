import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def register_activity_type(self, domain, name, version, task_list=None, default_task_heartbeat_timeout=None, default_task_schedule_to_close_timeout=None, default_task_schedule_to_start_timeout=None, default_task_start_to_close_timeout=None, description=None):
    """
        Registers a new activity type along with its configuration
        settings in the specified domain.

        :type domain: string
        :param domain: The name of the domain in which this activity is
            to be registered.

        :type name: string
        :param name: The name of the activity type within the domain.

        :type version: string
        :param version: The version of the activity type.

        :type task_list: string
        :param task_list: If set, specifies the default task list to
            use for scheduling tasks of this activity type. This default
            task list is used if a task list is not provided when a task
            is scheduled through the schedule_activity_task Decision.

        :type default_task_heartbeat_timeout: string
        :param default_task_heartbeat_timeout: If set, specifies the
            default maximum time before which a worker processing a task
            of this type must report progress by calling
            RecordActivityTaskHeartbeat. If the timeout is exceeded, the
            activity task is automatically timed out. This default can be
            overridden when scheduling an activity task using the
            ScheduleActivityTask Decision. If the activity worker
            subsequently attempts to record a heartbeat or returns a
            result, the activity worker receives an UnknownResource
            fault. In this case, Amazon SWF no longer considers the
            activity task to be valid; the activity worker should clean up
            the activity task.no docs

        :type default_task_schedule_to_close_timeout: string
        :param default_task_schedule_to_close_timeout: If set,
            specifies the default maximum duration for a task of this
            activity type. This default can be overridden when scheduling
            an activity task using the ScheduleActivityTask Decision.no
            docs

        :type default_task_schedule_to_start_timeout: string
        :param default_task_schedule_to_start_timeout: If set,
            specifies the default maximum duration that a task of this
            activity type can wait before being assigned to a worker. This
            default can be overridden when scheduling an activity task
            using the ScheduleActivityTask Decision.

        :type default_task_start_to_close_timeout: string
        :param default_task_start_to_close_timeout: If set, specifies
            the default maximum duration that a worker can take to process
            tasks of this activity type. This default can be overridden
            when scheduling an activity task using the
            ScheduleActivityTask Decision.

        :type description: string
        :param description: A textual description of the activity type.

        :raises: SWFTypeAlreadyExistsError, SWFLimitExceededError,
            UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('RegisterActivityType', {'domain': domain, 'name': name, 'version': version, 'defaultTaskList': {'name': task_list}, 'defaultTaskHeartbeatTimeout': default_task_heartbeat_timeout, 'defaultTaskScheduleToCloseTimeout': default_task_schedule_to_close_timeout, 'defaultTaskScheduleToStartTimeout': default_task_schedule_to_start_timeout, 'defaultTaskStartToCloseTimeout': default_task_start_to_close_timeout, 'description': description})