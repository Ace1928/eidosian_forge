import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_validate_mapping_structure_schema2_0(self):
    utils.validate_mapping_structure(self.attribute_mapping_schema_2_0)