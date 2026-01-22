from heat.common import exception
from heat.engine import properties_group as pg
from heat.tests import common
def test_properties_group_schema_validate(self):
    if self.message is not None:
        ex = self.assertRaises(exception.InvalidSchemaError, pg.PropertiesGroup, self.schema)
        self.assertEqual(self.message, str(ex))
    else:
        self.assertIsInstance(pg.PropertiesGroup(self.schema), pg.PropertiesGroup)