import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def poll_for_task(self, worker_group, hostname=None, instance_identity=None):
    """
        Task runners call this action to receive a task to perform
        from AWS Data Pipeline. The task runner specifies which tasks
        it can perform by setting a value for the workerGroup
        parameter of the PollForTask call. The task returned by
        PollForTask may come from any of the pipelines that match the
        workerGroup value passed in by the task runner and that was
        launched using the IAM user credentials specified by the task
        runner.

        If tasks are ready in the work queue, PollForTask returns a
        response immediately. If no tasks are available in the queue,
        PollForTask uses long-polling and holds on to a poll
        connection for up to a 90 seconds during which time the first
        newly scheduled task is handed to the task runner. To
        accomodate this, set the socket timeout in your task runner to
        90 seconds. The task runner should not call PollForTask again
        on the same `workerGroup` until it receives a response, and
        this may take up to 90 seconds.

        :type worker_group: string
        :param worker_group: Indicates the type of task the task runner is
            configured to accept and process. The worker group is set as a
            field on objects in the pipeline when they are created. You can
            only specify a single value for `workerGroup` in the call to
            PollForTask. There are no wildcard values permitted in
            `workerGroup`, the string must be an exact, case-sensitive, match.

        :type hostname: string
        :param hostname: The public DNS name of the calling task runner.

        :type instance_identity: dict
        :param instance_identity: Identity information for the Amazon EC2
            instance that is hosting the task runner. You can get this value by
            calling the URI, `http://169.254.169.254/latest/meta-data/instance-
            id`, from the EC2 instance. For more information, go to `Instance
            Metadata`_ in the Amazon Elastic Compute Cloud User Guide. Passing
            in this value proves that your task runner is running on an EC2
            instance, and ensures the proper AWS Data Pipeline service charges
            are applied to your pipeline.

        """
    params = {'workerGroup': worker_group}
    if hostname is not None:
        params['hostname'] = hostname
    if instance_identity is not None:
        params['instanceIdentity'] = instance_identity
    return self.make_request(action='PollForTask', body=json.dumps(params))