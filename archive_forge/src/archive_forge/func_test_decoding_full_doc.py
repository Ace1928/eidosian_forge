from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
def test_decoding_full_doc(self):
    """Simple List decoding that had caused some errors"""
    dynamizer = types.Dynamizer()
    doc = '{"__type__":{"S":"Story"},"company_tickers":{"SS":["NASDAQ-TSLA","NYSE-F","NYSE-GM"]},"modified_at":{"N":"1452525162"},"created_at":{"N":"1452525162"},"version":{"N":"1"},"categories":{"SS":["AUTOMTVE","LTRTR","MANUFCTU","PN","PRHYPE","TAXE","TJ","TL"]},"provider_categories":{"L":[{"S":"F"},{"S":"GM"},{"S":"TSLA"}]},"received_at":{"S":"2016-01-11T11:26:31Z"}}'
    output_doc = {'provider_categories': ['F', 'GM', 'TSLA'], '__type__': 'Story', 'company_tickers': set(['NASDAQ-TSLA', 'NYSE-GM', 'NYSE-F']), 'modified_at': Decimal('1452525162'), 'version': Decimal('1'), 'received_at': '2016-01-11T11:26:31Z', 'created_at': Decimal('1452525162'), 'categories': set(['LTRTR', 'TAXE', 'MANUFCTU', 'TL', 'TJ', 'AUTOMTVE', 'PRHYPE', 'PN'])}
    self.assertEqual(json.loads(doc, object_hook=dynamizer.decode), output_doc)