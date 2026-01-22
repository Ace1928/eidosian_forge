from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def overwatch_attribute_config():
    return concepts.ResourceParameterAttributeConfig(name='overwatch', help_text='Overwatch ID of the {resource}.')