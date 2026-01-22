from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.machine_images import flags
def construct_requests(client, machine_image_refs):
    requests = []
    for machine_image_ref in machine_image_refs:
        delete_request = (client.apitools_client.machineImages, 'Delete', client.messages.ComputeMachineImagesDeleteRequest(**machine_image_ref.AsDict()))
        requests.append(delete_request)
    return requests