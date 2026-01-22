import nltk
import os
import re
import itertools
import collections
import pkg_resources
from rouge import Rouge


class RougeAugmented(Rouge):
    def __init__(self, metrics=None, max_n=None, limit_length=True, length_limit=665, length_limit_type='bytes', apply_avg=True, apply_best=False, stemming=True, alpha=0.5, weight_factor=1.0, ensure_compatibility=True):
        """
        Handle the ROUGE score computation as in the official perl script.

        Note 1: Small differences might happen if the resampling of the perl script is not high enough (as the average depends on this).
        Note 2: Stemming of the official Porter Stemmer of the ROUGE perl script is slightly different and the Porter one implemented in NLTK. However, special cases of DUC 2004 have been traited.
                The solution would be to rewrite the whole perl stemming in python from the original script

        Args:
          metrics: What ROUGE score to compute. Available: ROUGE-N, ROUGE-L, ROUGE-W. Default: ROUGE-N
          max_n: N-grams for ROUGE-N if specify. Default:1
          limit_length: If the summaries must be truncated. Defaut:True
          length_limit: Number of the truncation where the unit is express int length_limit_Type. Default:665 (bytes)
          length_limit_type: Unit of length_limit. Available: words, bytes. Default: 'bytes'
          apply_avg: If we should average the score of multiple samples. Default: True. If apply_Avg & apply_best = False, then each ROUGE scores are independant
          apply_best: Take the best instead of the average. Default: False, then each ROUGE scores are independant
          stemming: Apply stemming to summaries. Default: True
          alpha: Alpha use to compute f1 score: P*R/((1-a)*P + a*R). Default:0.5
          weight_factor: Weight factor to be used for ROUGE-W. Official rouge score defines it at 1.2. Default: 1.0
          ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough). Default:True

        Raises:
          ValueError: raises exception if metric is not among AVAILABLE_METRICS
          ValueError: raises exception if length_limit_type is not among AVAILABLE_LENGTH_LIMIT_TYPES
          ValueError: raises exception if weight_factor < 0
        """
        super().__init__(Rouge)

    def get_scores(self, hypothesis, references):
        """
        Compute precision, recall and f1 score between hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between hypothesis and references

        Raises:
          ValueError: raises exception if a type of hypothesis is different than the one of reference
          ValueError: raises exception if a len of hypothesis is different than the one of reference
        """
        scores_default_rouge = Rouge.get_scores(self, hypothesis, references)

        scores = scores_default_rouge
        return scores
