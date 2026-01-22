import time
import logging
import pickle
import functools
import warnings
from packaging import version
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray.air.constants import TRAINING_ITERATION
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict, validate_warmstart
A wrapper around Optuna to provide trial suggestions.

    `Optuna <https://optuna.org/>`_ is a hyperparameter optimization library.
    In contrast to other libraries, it employs define-by-run style
    hyperparameter definitions.

    This Searcher is a thin wrapper around Optuna's search algorithms.
    You can pass any Optuna sampler, which will be used to generate
    hyperparameter suggestions.

    Multi-objective optimization is supported.

    Args:
        space: Hyperparameter search space definition for
            Optuna's sampler. This can be either a :class:`dict` with
            parameter names as keys and ``optuna.distributions`` as values,
            or a Callable - in which case, it should be a define-by-run
            function using ``optuna.trial`` to obtain the hyperparameter
            values. The function should return either a :class:`dict` of
            constant values with names as keys, or None.
            For more information, see https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html.

            .. warning::
                No actual computation should take place in the define-by-run
                function. Instead, put the training logic inside the function
                or class trainable passed to ``tune.Tuner()``.

        metric: The training result objective value attribute. If
            None but a mode was passed, the anonymous metric ``_metric``
            will be used per default. Can be a list of metrics for
            multi-objective optimization.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute. Can be a list of
            modes for multi-objective optimization (corresponding to
            ``metric``).
        points_to_evaluate: Initial parameter suggestions to be run
            first. This is for when you already have some good parameters
            you want to run first to help the algorithm make better suggestions
            for future parameters. Needs to be a list of dicts containing the
            configurations.
        sampler: Optuna sampler used to
            draw hyperparameter configurations. Defaults to ``MOTPESampler``
            for multi-objective optimization with Optuna<2.9.0, and
            ``TPESampler`` in every other case.
            See https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
            for available Optuna samplers.

            .. warning::
                Please note that with Optuna 2.10.0 and earlier
                default ``MOTPESampler``/``TPESampler`` suffer
                from performance issues when dealing with a large number of
                completed trials (approx. >100). This will manifest as
                a delay when suggesting new configurations.
                This is an Optuna issue and may be fixed in a future
                Optuna release.

        seed: Seed to initialize sampler with. This parameter is only
            used when ``sampler=None``. In all other cases, the sampler
            you pass should be initialized with the seed already.
        evaluated_rewards: If you have previously evaluated the
            parameters passed in as points_to_evaluate you can avoid
            re-running those trials by passing in the reward attributes
            as a list so the optimiser can be told the results without
            needing to re-compute the trial. Must be the same length as
            points_to_evaluate.

            .. warning::
                When using ``evaluated_rewards``, the search space ``space``
                must be provided as a :class:`dict` with parameter names as
                keys and ``optuna.distributions`` instances as values. The
                define-by-run search space definition is not yet supported with
                this functionality.

    Tune automatically converts search spaces to Optuna's format:

    .. code-block:: python

        from ray.tune.search.optuna import OptunaSearch

        config = {
            "a": tune.uniform(6, 8)
            "b": tune.loguniform(1e-4, 1e-2)
        }

        optuna_search = OptunaSearch(
            metric="loss",
            mode="min")

        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
            ),
            param_space=config,
        )
        tuner.fit()

    If you would like to pass the search space manually, the code would
    look like this:

    .. code-block:: python

        from ray.tune.search.optuna import OptunaSearch
        import optuna

        space = {
            "a": optuna.distributions.FloatDistribution(6, 8),
            "b": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
        }

        optuna_search = OptunaSearch(
            space,
            metric="loss",
            mode="min")

        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
            ),
        )
        tuner.fit()

        # Equivalent Optuna define-by-run function approach:

        def define_search_space(trial: optuna.Trial):
            trial.suggest_float("a", 6, 8)
            trial.suggest_float("b", 1e-4, 1e-2, log=True)
            # training logic goes into trainable, this is just
            # for search space definition

        optuna_search = OptunaSearch(
            define_search_space,
            metric="loss",
            mode="min")

        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
            ),
        )
        tuner.fit()

    Multi-objective optimization is supported:

    .. code-block:: python

        from ray.tune.search.optuna import OptunaSearch
        import optuna

        space = {
            "a": optuna.distributions.FloatDistribution(6, 8),
            "b": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
        }

        # Note you have to specify metric and mode here instead of
        # in tune.TuneConfig
        optuna_search = OptunaSearch(
            space,
            metric=["loss1", "loss2"],
            mode=["min", "max"])

        # Do not specify metric and mode here!
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
            ),
        )
        tuner.fit()

    You can pass configs that will be evaluated first using
    ``points_to_evaluate``:

    .. code-block:: python

        from ray.tune.search.optuna import OptunaSearch
        import optuna

        space = {
            "a": optuna.distributions.FloatDistribution(6, 8),
            "b": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
        }

        optuna_search = OptunaSearch(
            space,
            points_to_evaluate=[{"a": 6.5, "b": 5e-4}, {"a": 7.5, "b": 1e-3}]
            metric="loss",
            mode="min")

        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
            ),
        )
        tuner.fit()

    Avoid re-running evaluated trials by passing the rewards together with
    `points_to_evaluate`:

    .. code-block:: python

        from ray.tune.search.optuna import OptunaSearch
        import optuna

        space = {
            "a": optuna.distributions.FloatDistribution(6, 8),
            "b": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
        }

        optuna_search = OptunaSearch(
            space,
            points_to_evaluate=[{"a": 6.5, "b": 5e-4}, {"a": 7.5, "b": 1e-3}]
            evaluated_rewards=[0.89, 0.42]
            metric="loss",
            mode="min")

        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
            ),
        )
        tuner.fit()

    .. versionadded:: 0.8.8

    