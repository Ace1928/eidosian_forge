from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def get_parent_resource_specs():
    return concepts.ResourceSpec('securedlandingzone.organizations.locations', resource_name='parent', organizationsId=organization_attribute_config(), locationsId=location_attribute_config())