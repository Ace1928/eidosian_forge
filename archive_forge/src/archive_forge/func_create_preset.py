from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def create_preset(self, name=None, description=None, container=None, video=None, audio=None, thumbnails=None):
    """
        The CreatePreset operation creates a preset with settings that
        you specify.
        Elastic Transcoder checks the CreatePreset settings to ensure
        that they meet Elastic Transcoder requirements and to
        determine whether they comply with H.264 standards. If your
        settings are not valid for Elastic Transcoder, Elastic
        Transcoder returns an HTTP 400 response (
        `ValidationException`) and does not create the preset. If the
        settings are valid for Elastic Transcoder but aren't strictly
        compliant with the H.264 standard, Elastic Transcoder creates
        the preset and returns a warning message in the response. This
        helps you determine whether your settings comply with the
        H.264 standard while giving you greater flexibility with
        respect to the video that Elastic Transcoder produces.
        Elastic Transcoder uses the H.264 video-compression format.
        For more information, see the International Telecommunication
        Union publication Recommendation ITU-T H.264: Advanced video
        coding for generic audiovisual services .

        :type name: string
        :param name: The name of the preset. We recommend that the name be
            unique within the AWS account, but uniqueness is not enforced.

        :type description: string
        :param description: A description of the preset.

        :type container: string
        :param container: The container type for the output file. Valid values
            include `mp3`, `mp4`, `ogg`, `ts`, and `webm`.

        :type video: dict
        :param video: A section of the request body that specifies the video
            parameters.

        :type audio: dict
        :param audio: A section of the request body that specifies the audio
            parameters.

        :type thumbnails: dict
        :param thumbnails: A section of the request body that specifies the
            thumbnail parameters, if any.

        """
    uri = '/2012-09-25/presets'
    params = {}
    if name is not None:
        params['Name'] = name
    if description is not None:
        params['Description'] = description
    if container is not None:
        params['Container'] = container
    if video is not None:
        params['Video'] = video
    if audio is not None:
        params['Audio'] = audio
    if thumbnails is not None:
        params['Thumbnails'] = thumbnails
    return self.make_request('POST', uri, expected_status=201, data=json.dumps(params))