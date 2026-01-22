from pyperf import Runner
from referencing import Registry
from referencing.jsonschema import DRAFT201909
from jsonschema import Draft201909Validator

An unused schema registry should not cause slower validation.

"Unused" here means one where no reference resolution is occurring anyhow.

See https://github.com/python-jsonschema/jsonschema/issues/1088.
