from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def dataset_config_attribute_config():
    return concepts.ResourceParameterAttributeConfig(name='dataset-config', help_text='Dataset Config ID for the {resource}.')