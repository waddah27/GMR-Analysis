"""
Extended Gaussian Mixture Model (GMM) for clustering data.

This implementation extends the sklearn GaussianMixture class to include
the functionality of fitting the model parameters and predicting the labels
for the input data. In addition, it stores the history of the log likelihood
and lower bound during the fitting process.
"""

import warnings
import numpy as np
from typing import Any, Literal
from matplotlib.pylab import RandomState
from numpy import ndarray
from sklearn.dummy import check_random_state
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

class GMM_EXTENDED(GaussianMixture):
    """
    Extended Gaussian Mixture Model (GMM) for clustering data.

    This implementation extends the sklearn GaussianMixture class to include
    the functionality of fitting the model parameters and predicting the labels
    for the input data. In addition, it stores the history of the log likelihood
    and lower bound during the fitting process.

    Parameters
    ----------
    n_components : int, optional
        The number of components in the model. Default is 1.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, optional
        The type of covariance matrix to use. Default is 'full'.
    tol : float, optional
        The convergence tolerance for the fitting process. Default is 0.001.
    reg_covar : float, optional
        The regularization parameter for the covariance matrix. Default is 0.000001.
    max_iter : int, optional
        The maximum number of iterations for the fitting process. Default is 100.
    n_init : int, optional
        The number of initialisation trials for the fitting process. Default is 1.
    init_params : {'kmeans', 'random'}, optional
        The method to initialise the model parameters. Default is 'kmeans'.
    weights_init : array-like, shape (n_components,), optional
        The initial weights for the model. Default is None.
    means_init : array-like, shape (n_components, n_features), optional
        The initial means for the model. Default is None.
    precisions_init : array-like, shape (n_components, n_features), optional
        The initial precisions for the model. Default is None.
    random_state : int, RandomState instance or None, optional
        The random state for the initialisation process. Default is None.
    warm_start : bool, optional
        Whether to use the previous model parameters as initialisation. Default is False.
    verbose : int, optional
        The level of verbosity for the fitting process. Default is 0.
    verbose_interval : int, optional
        The interval for printing the fitting progress. Default is 10.
    """

    def __init__(self, n_components = 1,*, covariance_type = "full", tol = 0.001, reg_covar = 0.000001, max_iter = 100,
                 n_init = 1, init_params = "kmeans", weights_init= None, means_init = None, precisions_init = None,
                 random_state = None, warm_start = False, verbose = 0, verbose_interval = 10) -> None:
        super().__init__(n_components, covariance_type=covariance_type, tol=tol, reg_covar=reg_covar, max_iter=max_iter,
                         n_init=n_init, init_params=init_params, weights_init=weights_init, means_init=means_init,
                         precisions_init=precisions_init, random_state=random_state, warm_start=warm_start, verbose=verbose,
                         verbose_interval=verbose_interval)
        self.dlog_history = []
        self.log_history = []
        self.llog_hist = []
        self.log_resp = []
        self.scaler = 1

    def fit_predict(self, X, y=None):
        """
        Estimate model parameters using X and predict the labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        # rest of the code remains the same...
