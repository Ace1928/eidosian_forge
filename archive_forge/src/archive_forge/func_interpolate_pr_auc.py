from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def interpolate_pr_auc(tp, fp, fn):
    """Interpolation formula inspired by section 4 of (Davis et al., 2006).

      Note here we derive & use a closed formula not present in the paper
      - as follows:
      Modeling all of TP (true positive weight),
      FP (false positive weight) and their sum P = TP + FP (positive weight)
      as varying linearly within each interval [A, B] between successive
      thresholds, we get
        Precision = (TP_A + slope * (P - P_A)) / P
      with slope = dTP / dP = (TP_B - TP_A) / (P_B - P_A).
      The area within the interval is thus (slope / total_pos_weight) times
        int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
        int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}
      where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in
        int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)
      Bringing back the factor (slope / total_pos_weight) we'd put aside, we get
         slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight
      where dTP == TP_B - TP_A.
      Note that when P_A == 0 the above calculation simplifies into
        int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)
      which is really equivalent to imputing constant precision throughout the
      first bucket having >0 true positives.

      Args:
        tp: true positive counts
        fp: false positive counts
        fn: false negative counts

      Returns:
        pr_auc: an approximation of the area under the P-R curve.

      References:
        The Relationship Between Precision-Recall and ROC Curves:
          [Davis et al., 2006](https://dl.acm.org/citation.cfm?id=1143874)
          ([pdf](https://www.biostat.wisc.edu/~page/rocpr.pdf))
      """
    dtp = tp[:num_thresholds - 1] - tp[1:]
    p = tp + fp
    prec_slope = math_ops.div_no_nan(dtp, math_ops.maximum(p[:num_thresholds - 1] - p[1:], 0), name='prec_slope')
    intercept = tp[1:] - math_ops.multiply(prec_slope, p[1:])
    safe_p_ratio = array_ops.where(math_ops.logical_and(p[:num_thresholds - 1] > 0, p[1:] > 0), math_ops.div_no_nan(p[:num_thresholds - 1], math_ops.maximum(p[1:], 0), name='recall_relative_ratio'), array_ops.ones_like(p[1:]))
    return math_ops.reduce_sum(math_ops.div_no_nan(prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)), math_ops.maximum(tp[1:] + fn[1:], 0), name='pr_auc_increment'), name='interpolate_pr_auc')