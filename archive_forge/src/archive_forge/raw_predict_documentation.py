from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import sys
from googlecloudsdk.api_lib.ai.endpoints import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
Run Vertex AI online raw prediction.

  `{command}` sends a raw prediction request to a Vertex AI endpoint. The
  request can be given on the command line or read from a file or stdin.

  ## EXAMPLES

  To predict against an endpoint ``123'' under project ``example'' in region
  ``us-central1'', reading the request from the command line, run:

    $ {command} 123 --project=example --region=us-central1 --request='{
        "instances": [
          { "values": [1, 2, 3, 4], "key": 1 },
          { "values": [5, 6, 7, 8], "key": 2 }
        ]
      }'

  If the request body was in the file ``input.json'', run:

    $ {command} 123 --project=example --region=us-central1 --request=@input.json

  To send the image file ``image.jpeg'' and set the *content type*, run:

    $ {command} 123 --project=example --region=us-central1
    --http-headers=Content-Type=image/jpeg --request=@image.jpeg
  