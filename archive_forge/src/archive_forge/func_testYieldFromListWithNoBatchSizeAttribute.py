import unittest
from apitools.base.py import list_pager
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 \
from samples.fusiontables_sample.fusiontables_v1 \
from samples.iam_sample.iam_v1 import iam_v1_client as iam_client
from samples.iam_sample.iam_v1 import iam_v1_messages as iam_messages
def testYieldFromListWithNoBatchSizeAttribute(self):
    self.mocked_client.iamPolicies.GetPolicyDetails.Expect(iam_messages.GetPolicyDetailsRequest(pageToken=None, fullResourcePath='myresource'), iam_messages.GetPolicyDetailsResponse(policies=[iam_messages.PolicyDetail(fullResourcePath='c0'), iam_messages.PolicyDetail(fullResourcePath='c1')]))
    client = iam_client.IamV1(get_credentials=False)
    request = iam_messages.GetPolicyDetailsRequest(fullResourcePath='myresource')
    results = list_pager.YieldFromList(client.iamPolicies, request, batch_size_attribute=None, method='GetPolicyDetails', field='policies')
    i = 0
    for i, instance in enumerate(results):
        self.assertEquals('c{0}'.format(i), instance.fullResourcePath)
    self.assertEquals(1, i)