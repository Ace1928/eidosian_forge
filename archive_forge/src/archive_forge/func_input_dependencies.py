from typing import Iterator, List
@property
def input_dependencies(self) -> List['Operator']:
    """List of operators that provide inputs for this operator."""
    assert hasattr(self, '_input_dependencies'), 'Operator.__init__() was not called.'
    return self._input_dependencies