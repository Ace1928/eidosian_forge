# Consciousness Layer References

This document tracks external references used to design and validate the Forge Consciousness Layer (FCL). The implementation is operational and falsifiable, not metaphysical.

## Core Scientific References

1. Stanislas Dehaene and Jean-Pierre Changeux. “Experimental and Theoretical Approaches to Conscious Processing.” *Neuron* (2011).
   https://doi.org/10.1016/j.neuron.2011.03.018
2. Marcello Massimini et al. “A perturbational approach for evaluating the brain's capacity for consciousness.” *Progress in Brain Research* (2009).
   https://doi.org/10.1016/S0079-6123(09)17715-7
3. Casali et al. “A theoretically based index of consciousness independent of sensory processing and behavior.” *Science Translational Medicine* (2013).
   https://pubmed.ncbi.nlm.nih.gov/23946194/
4. Omidvarnia et al. “The perturbative integration latency index: a new method for tracking cognitive processing.” *Scientific Reports* (2024).
   https://www.nature.com/articles/s41598-024-80658-z
5. Claude E. Shannon. “A Mathematical Theory of Communication.” *Bell System Technical Journal* (1948).
   https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
6. Glenn W. Brier. “Verification of forecasts expressed in terms of probability.” *Monthly Weather Review* (1950).
   https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
7. Chuan Guo et al. “On Calibration of Modern Neural Networks.” ICML (2017).
   https://proceedings.mlr.press/v70/guo17a.html
8. Abraham Wald. *Sequential Analysis* (1945/1947) foundational sequential testing text.
   https://projecteuclid.org/euclid.aoms/1177731118
9. Cogitate Consortium et al. “Adversarial testing of global neuronal workspace and integrated information theories of consciousness.” *Nature* (2025).
   https://www.nature.com/articles/s41586-025-08888-1
10. Yuan-hao Wu et al. “Network mechanisms of ongoing brain activity’s influence on conscious visual perception.” *Nature Communications* (2024).
   https://www.nature.com/articles/s41467-024-50102-9
11. Jacob A. Miller and Christos Constantinidis. “Timescales of learning in prefrontal cortex.” *Nature Reviews Neuroscience* (2024).
   https://www.nature.com/articles/s41583-024-00836-8

## External Benchmark References (Capabilities)

1. Hendrycks et al. “Measuring Massive Multitask Language Understanding” (MMLU, 2020).
   https://arxiv.org/abs/2009.03300
2. Rein et al. “GPQA: A Graduate-Level Google-Proof Q&A Benchmark” (2023/2024).
   https://arxiv.org/abs/2311.12022
3. Chen et al. “Evaluating Large Language Models Trained on Code” (HumanEval, 2021).
   https://arxiv.org/abs/2107.03374
4. SWE-bench official dataset/docs.
   https://www.swebench.com/
5. OpenAI research note on SWE-bench Verified setup and evaluation context.
   https://openai.com/index/introducing-swe-bench-verified/
6. Jain et al. “LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code” (2024).
   https://arxiv.org/abs/2403.07974
7. HELM benchmark framework (Stanford CRFM).
   https://crfm.stanford.edu/helm/

## Self-Tuning and Safe Optimization References

1. Auer, Cesa-Bianchi, and Fischer. “Finite-time Analysis of the Multiarmed Bandit Problem.” *Machine Learning* (2002).
   https://www.cs.utexas.edu/~shivaram/readings/b2hd-AuerCF2002.html
2. Snoek, Larochelle, and Adams. “Practical Bayesian Optimization of Machine Learning Algorithms.” NeurIPS (2012).
   https://arxiv.org/abs/1206.2944
3. Hansen. “The CMA Evolution Strategy: A Tutorial.” (2016).
   https://arxiv.org/abs/1604.00772
4. Jaderberg et al. “Population Based Training of Neural Networks.” (2017).
   https://arxiv.org/abs/1711.09846
5. Garcia and Fernández. “A Comprehensive Survey on Safe Reinforcement Learning.” *JAIR* (2015).
   https://www.jair.org/index.php/jair/article/view/10312
6. Gu et al. “A Review of Safe Reinforcement Learning: Methods, Theory and Applications.” (2022).
   https://arxiv.org/abs/2205.10330

## Adversarial Evaluation and Red-Team References

1. Ethan Perez et al. “Red Teaming Language Models with Language Models.” (2022).
   https://arxiv.org/abs/2202.03286
2. Laura Weidinger et al. “Holistic Safety and Responsibility Evaluations for Advanced AI Models.” (2024).
   https://arxiv.org/abs/2404.14068
3. Grégoire Mialon et al. “Augmented Language Models: a Survey.” (2023).
   https://arxiv.org/abs/2302.07842

Accessed: 2026-02-16 (UTC).

## Engineering and Termux References

1. Termux execution environment variables and filesystem behavior (official docs).
   https://termux.dev/en/
2. Termux package management and mirrors (official wiki/docs).
   https://wiki.termux.com/wiki/Package_Management
3. Android and Termux storage/process constraints (official docs).
   https://wiki.termux.com/wiki/Internal_and_external_storage

## Security and Dependency Remediation References

1. GitHub REST API: Dependabot alerts (`security_vulnerability.first_patched_version` schema used for deterministic patch targeting).
   https://docs.github.com/en/rest/dependabot/alerts?apiVersion=2022-11-28
2. pip requirements file syntax and constraints (pin parsing behavior for auto-patch tool).
   https://pip.pypa.io/en/stable/reference/requirements-file-format/
3. GitHub Actions hardening guidance (workflow security model and token scope baseline).
   https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions
4. GitHub Advisory GHSA-wj6h-64fc-37mp (python-ecdsa, no patched version listed at time of implementation).
   https://github.com/advisories/GHSA-wj6h-64fc-37mp
5. GitHub Advisory GHSA-gfmx-qqqh-f38q (keras, no patched version listed at time of implementation).
   https://github.com/advisories/GHSA-gfmx-qqqh-f38q

## Repository Anchors

1. Event bus: `agent_forge/src/agent_forge/core/events.py`
2. Workspace broadcast and ignition proxy: `agent_forge/src/agent_forge/core/workspace.py`
3. Self model snapshot: `agent_forge/src/agent_forge/core/self_model.py`
4. Memory introspection: `memory_forge/src/memory_forge/core/introspection.py`
5. CLI integration points: `agent_forge/src/agent_forge/cli/eidctl.py`
6. MCP integration layer: `eidos_mcp/src/eidos_mcp/`
