from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def location_attribute_config():
    return concepts.ResourceParameterAttributeConfig(name='location', help_text='Google Cloud location for the {resource}.')