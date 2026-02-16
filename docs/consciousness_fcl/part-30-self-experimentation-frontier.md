# Part 30: Self-Experimentation Frontier

## Goal

Move from threshold tuning to structured self-experimentation: adaptive parameter search, multi-objective optimization, and safe experiment design.

## Near-Term Upgrades

1. ~~Replace hill-climbing with Bayesian optimizer for expensive trial budgets.~~
Status: implemented as `bayes_pareto` acquisition path in autotune.
2. Add multi-objective selection (coherence, trace strength, ownership, groundedness, stability).
3. ~~Add adaptive attention/competition weights trained from trial rewards.~~
Status: implemented with online trace-feedback learning loops in `attention` and `workspace_competition`.
4. Add experiment-designer module that proposes perturbation recipes from uncertainty gaps.
5. Add adversarial red-team campaigns as regression gates.

## Safety Envelope

- Keep dangerous parameters out of autonomous tuning set.
- Enforce hard guardrails before any commit.
- Persist every proposal and result with seed/spec/overlay hashes.
- Use automatic rollback for any regression or guardrail breach.

## Proposed New Modules

- `consciousness/modules/experiment_designer.py`
- `consciousness/tuning/bayes_optimizer.py`
- `consciousness/bench/red_team.py`
- `consciousness/bench/hypothesis_registry.py`

## Research Anchors

- Upper Confidence Bound bandits (Auer et al. 2002):  
  https://www.cs.utexas.edu/~shivaram/readings/b2hd-AuerCF2002.html
- Bayesian optimization for expensive functions (Snoek et al. 2012):  
  https://arxiv.org/abs/1206.2944
- CMA-ES strategy parameter adaptation (Hansen 2016):  
  https://arxiv.org/abs/1604.00772
- Population Based Training (Jaderberg et al. 2017):  
  https://arxiv.org/abs/1711.09846
- Safe reinforcement learning survey (García and Fernández 2015):  
  https://www.jair.org/index.php/jair/article/view/10312
- Safe RL overview and recent methods (Gu et al. 2022):  
  https://arxiv.org/abs/2205.10330

## Acceptance Criteria

1. Optimizer upgrades outperform hill-climbing on fixed trial budgets.
2. Multi-objective tuning avoids single-attractor collapse.
3. Experiment designer proposals increase information gain without safety violations.
4. Red-team suite catches confabulation/fake-ignition regressions pre-commit.
