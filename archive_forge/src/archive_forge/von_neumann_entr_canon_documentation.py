from cvxpy import Variable, lambda_sum_largest, trace
from cvxpy.atoms.affine.sum import sum
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.constraints.zero import Zero
from cvxpy.reductions.dcp2cone.canonicalizers.entr_canon import entr_canon
from cvxpy.reductions.dcp2cone.canonicalizers.lambda_sum_largest_canon import (

Copyright 2022, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
