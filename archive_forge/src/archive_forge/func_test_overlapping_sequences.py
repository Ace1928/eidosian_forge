import re
from unittest import mock
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_overlapping_sequences(self):
    template_rs = '\nHeatTemplateFormatVersion: \'2012-12-12\'\nResources:\n  secret:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 60\n      character_classes: []\n      character_sequences:\n        - sequence: "01"\n        - sequence: "02"\n        - sequence: "03"\n        - sequence: "04"\n        - sequence: "05"\n        - sequence: "06"\n        - sequence: "07"\n        - sequence: "08"\n        - sequence: "09"\n'
    results = self.run_test(template_rs)
    self.check_stats(self.char_counts(results, '0'), 10, 5)