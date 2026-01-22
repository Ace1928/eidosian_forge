import unittest
from apitools.base.py import extra_types
from samples.servicemanagement_sample.servicemanagement_v1 \
def testInstantiateMessageWithAdditionalProperties(self):
    PROJECT_NAME = 'test-project'
    SERVICE_NAME = 'test-service'
    SERVICE_VERSION = '1.0'
    prop = messages.Operation.ResponseValue.AdditionalProperty
    messages.Operation(name='operation-12345-67890', done=False, response=messages.Operation.ResponseValue(additionalProperties=[prop(key='producerProjectId', value=extra_types.JsonValue(string_value=PROJECT_NAME)), prop(key='serviceName', value=extra_types.JsonValue(string_value=SERVICE_NAME)), prop(key='serviceConfig', value=extra_types.JsonValue(object_value=extra_types.JsonObject(properties=[extra_types.JsonObject.Property(key='id', value=extra_types.JsonValue(string_value=SERVICE_VERSION))])))]))