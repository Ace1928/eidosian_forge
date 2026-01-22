from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def update_pipeline_notifications(self, id=None, notifications=None):
    """
        With the UpdatePipelineNotifications operation, you can update
        Amazon Simple Notification Service (Amazon SNS) notifications
        for a pipeline.

        When you update notifications for a pipeline, Elastic
        Transcoder returns the values that you specified in the
        request.

        :type id: string
        :param id: The identifier of the pipeline for which you want to change
            notification settings.

        :type notifications: dict
        :param notifications:
        The topic ARN for the Amazon Simple Notification Service (Amazon SNS)
            topic that you want to notify to report job status.
        To receive notifications, you must also subscribe to the new topic in
            the Amazon SNS console.

        + **Progressing**: The topic ARN for the Amazon Simple Notification
              Service (Amazon SNS) topic that you want to notify when Elastic
              Transcoder has started to process jobs that are added to this
              pipeline. This is the ARN that Amazon SNS returned when you created
              the topic.
        + **Completed**: The topic ARN for the Amazon SNS topic that you want
              to notify when Elastic Transcoder has finished processing a job.
              This is the ARN that Amazon SNS returned when you created the
              topic.
        + **Warning**: The topic ARN for the Amazon SNS topic that you want to
              notify when Elastic Transcoder encounters a warning condition. This
              is the ARN that Amazon SNS returned when you created the topic.
        + **Error**: The topic ARN for the Amazon SNS topic that you want to
              notify when Elastic Transcoder encounters an error condition. This
              is the ARN that Amazon SNS returned when you created the topic.

        """
    uri = '/2012-09-25/pipelines/{0}/notifications'.format(id)
    params = {}
    if id is not None:
        params['Id'] = id
    if notifications is not None:
        params['Notifications'] = notifications
    return self.make_request('POST', uri, expected_status=200, data=json.dumps(params))