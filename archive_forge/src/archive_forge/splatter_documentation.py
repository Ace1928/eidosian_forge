from . import r_function
import numbers
import numpy as np
import warnings
Simulate count data from a fictional single-cell RNA-seq experiment Splat.

    SplatSimulate is a Python wrapper for the R package Splatter. For more
    details, read about Splatter on GitHub_ and Bioconductor_.

    .. _GitHub: https://github.com/Oshlack/splatter
    .. _Bioconductor: https://bioconductor.org/packages/release/bioc/html/splatter.html

    Parameters
    ----------
    batch_cells : list-like or int, optional (default: 100)
        The number of cells in each batch.
    n_genes : int, optional (default:10000)
        The number of genes to simulate.
    batch_fac_loc : float, optional (default: 0.1)
        Location (meanlog) parameter for the batch effects factor
        log-normal distribution.
    batch_fac_scale : float, optional (default: 0.1)
        Scale (sdlog) parameter for the batch effects factor
        log-normal distribution.
    mean_shape : float, optional (default: 0.3)
        Shape parameter for the mean gamma distribution.
    mean_rate : float, optional (default: 0.6)
        Rate parameter for the mean gamma distribution.
    lib_loc : float, optional (default: 11)
        Location (meanlog) parameter for the library size
        log-normal distribution, or mean for the normal distribution.
    lib_scale : float, optional (default: 0.2)
        Scale (sdlog) parameter for the library size log-normal distribution,
        or sd for the normal distribution.
    lib_norm : bool, optional (default: False)
        Whether to use a normal distribution instead of the usual
        log-normal distribution.
    out_prob : float, optional (default: 0.05)
        Probability that a gene is an expression outlier.
    out_fac_loc : float, optional (default: 4)
        Location (meanlog) parameter for the expression outlier factor
        log-normal distribution.
    out_fac_scale : float, optional (default: 0.5)
        Scale (sdlog) parameter for the expression outlier factor
        log-normal distribution.
    de_prob : float, optional (default: 0.1)
        Probability that a gene is differentially expressed in each
        group or path.
    de_down_prob : float, optional (default: 0.1)
        Probability that a differentially expressed gene is down-regulated.
    de_fac_loc : float, optional (default: 0.1)
        Location (meanlog) parameter for the differential expression factor
        log-normal distribution.
    de_fac_scale : float, optional (default: 0.4)
        Scale (sdlog) parameter for the differential expression factor
        log-normal distribution.
    bcv_common : float, optional (default: 0.1)
        Underlying common dispersion across all genes.
    bcv_df float, optional (default: 60)
        Degrees of Freedom for the BCV inverse chi-squared distribution.
    dropout_type : {'none', 'experiment', 'batch', 'group', 'cell', 'binomial'},
        optional (default: 'none')
        The type of dropout to simulate. "none" indicates no dropout,
        "experiment" is global dropout using the same parameters for every
        cell, "batch" uses the same parameters for every cell in each batch,
        "group" uses the same parameters for every cell in each groups,
        "cell" uses a different set of parameters for each cell, and
        "binomial" performs post-hoc binomial undersampling.
    dropout_mid : list-like or float, optional (default: 0)
        Midpoint parameter for the dropout logistic function.
    dropout_shape : list-like or float, optional (default: -1)
        Shape parameter for the dropout logistic function.
    dropout_prob : float, optional (default: 0.5)
        Probability for binomial undersampling dropout.
    group_prob : list-like or int, optional (default: 1, shape=[n_groups])
        The probabilities that cells come from particular groups.
    path_from : list-like, optional (default: 0, shape=[n_groups])
        Vector giving the originating point of each path.
    path_length : list-like, optional (default: 100, shape=[n_groups])
        Vector giving the number of steps to simulate along each path.
    path_skew : list-like, optional (default: 0.5, shape=[n_groups])
        Vector giving the skew of each path.
    path_nonlinear_prob : float, optional (default: 0.1)
        Probability that a gene changes expression in a non-linear way along
        the differentiation path.
    path_sigma_fac : float, optional (default: 0.8)
        Sigma factor for non-linear gene paths.
    seed : int or None, optional (default: None)
        Seed to use for generating random numbers.
    verbose : int, optional (default: 1)
        Logging verbosity between 0 and 2.

    Returns
    -------
    sim : dict
        counts : Simulated expression counts.
        group : The group or path the cell belongs to.
        batch : The batch the cell was sampled from.
        exp_lib_size : The expected library size for that cell.
        step (paths only) : how far along the path each cell is.
        base_gene_mean : The base expression level for that gene.
        outlier_factor : Expression outlier factor for that gene. Values of 1 indicate
            the gene is not an expression outlier.
        gene_mean : Expression level after applying outlier factors.
        batch_fac_[batch] : The batch effects factor for each gene for a particular
            batch.
        de_fac_[group] : The differential expression factor for each gene in a
            particular group. Values of 1 indicate the gene is not differentially
            expressed.
        sigma_fac_[path] : Factor applied to genes that have non-linear changes in
            expression along a path.
        batch_cell_means : The mean expression of genes in each cell after adding
            batch effects.
        base_cell_means : The mean expression of genes in each cell after any
            differential expression and adjusted for expected library size.
        bcv : The Biological Coefficient of Variation for each gene in each cell.
        cell_means : The mean expression level of genes in each cell adjusted for BCV.
        true_counts : The simulated counts before dropout.
        dropout : Logical matrix showing which values have been dropped in which cells.
    