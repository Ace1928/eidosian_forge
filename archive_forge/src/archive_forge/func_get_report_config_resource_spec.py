from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def get_report_config_resource_spec():
    return concepts.ResourceSpec('storageinsights.projects.locations.reportConfigs', resource_name='report-config', reportConfigsId=report_config_attribute_config(), locationsId=location_attribute_config(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)