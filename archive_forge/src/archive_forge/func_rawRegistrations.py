import re
import torch._C as C
def rawRegistrations(self):
    return C._dispatch_dump(f'{self.namespace}::{self.name}')