from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_candidate_sampling_ops
from tensorflow.python.ops import math_ops  # pylint: disable=unused-import
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('random.fixed_unigram_candidate_sampler', 'nn.fixed_unigram_candidate_sampler')
@dispatch.add_dispatch_support
def fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, vocab_file='', distortion=1.0, num_reserved_ids=0, num_shards=1, shard=0, unigrams=(), seed=None, name=None):
    """Samples a set of classes using the provided (fixed) base distribution.

  This operation randomly samples a tensor of sampled classes
  (`sampled_candidates`) from the range of integers `[0, range_max)`.

  See the [Candidate Sampling Algorithms
  Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf)
  for a quick course on Candidate Sampling.

  The elements of `sampled_candidates` are drawn without replacement
  (if `unique=True`) or with replacement (if `unique=False`) from
  the base distribution.

  The base distribution is read from a file or passed in as an
  in-memory array. There is also an option to skew the distribution by
  applying a distortion power to the weights.

  In addition, this operation returns tensors `true_expected_count`
  and `sampled_expected_count` representing the number of times each
  of the target classes (`true_classes`) and the sampled
  classes (`sampled_candidates`) is expected to occur in an average
  tensor of sampled classes.  These values correspond to `Q(y|x)`
  defined in the [Candidate Sampling Algorithms
  Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf).
  If `unique=True`, then these are post-rejection probabilities and we
  compute them approximately.

  Note that this function (and also other `*_candidate_sampler`
  functions) only gives you the ingredients to implement the various
  Candidate Sampling algorithms listed in the big table in the
  [Candidate Sampling Algorithms
  Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf). You
  still need to implement the algorithms yourself.

  For example, according to that table, the phrase "negative samples"
  may mean different things in different algorithms. For instance, in
  NCE, "negative samples" means `S_i` (which is just the sampled
  classes) which may overlap with true classes, while in Sampled
  Logistic, "negative samples" means `S_i - T_i` which excludes the
  true classes. The return value `sampled_candidates` corresponds to
  `S_i`, not to any specific definition of "negative samples" in any
  specific algorithm. It's your responsibility to pick an algorithm
  and calculate the "negative samples" defined by that algorithm
  (e.g. `S_i - T_i`).

  As another example, the `true_classes` argument is for calculating
  the `true_expected_count` output (as a by-product of this function's
  main calculation), which may be needed by some algorithms (according
  to that table). It's not for excluding true classes in the return
  value `sampled_candidates`. Again that step is algorithm-specific
  and should be carried out by you.

  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of classes to randomly sample.
    unique: A `bool`. Determines whether all sampled classes in a batch are
      unique.
    range_max: An `int`. The number of possible classes.
    vocab_file: Each valid line in this file (which should have a CSV-like
      format) corresponds to a valid word ID. IDs are in sequential order,
      starting from num_reserved_ids. The last entry in each line is expected
      to be a value corresponding to the count or relative probability. Exactly
      one of `vocab_file` and `unigrams` needs to be passed to this operation.
    distortion: The distortion is used to skew the unigram probability
      distribution.  Each weight is first raised to the distortion's power
      before adding to the internal unigram distribution. As a result,
      `distortion = 1.0` gives regular unigram sampling (as defined by the vocab
      file), and `distortion = 0.0` gives a uniform distribution.
    num_reserved_ids: Optionally some reserved IDs can be added in the range
      `[0, num_reserved_ids)` by the users. One use case is that a special
      unknown word token is used as ID 0. These IDs will have a sampling
      probability of 0.
    num_shards: A sampler can be used to sample from a subset of the original
      range in order to speed up the whole computation through parallelism. This
      parameter (together with `shard`) indicates the number of partitions that
      are being used in the overall computation.
    shard: A sampler can be used to sample from a subset of the original range
      in order to speed up the whole computation through parallelism. This
      parameter (together with `num_shards`) indicates the particular partition
      number of the operation, when partitioning is being used.
    unigrams: A list of unigram counts or probabilities, one per ID in
      sequential order. Exactly one of `vocab_file` and `unigrams` should be
      passed to this operation.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).

  Returns:
    sampled_candidates: A tensor of type `int64` and shape
      `[num_sampled]`. The sampled classes. As noted above,
      `sampled_candidates` may overlap with true classes.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`.

  """
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_candidate_sampling_ops.fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, vocab_file=vocab_file, distortion=distortion, num_reserved_ids=num_reserved_ids, num_shards=num_shards, shard=shard, unigrams=unigrams, seed=seed1, seed2=seed2, name=name)