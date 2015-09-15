# -*- coding: utf-8 -*-
r"""
===============================================================================
                Minimax Inference for Linear Systems 
===============================================================================

This module implements the minimax filter as described by Dan Simon (2006). 
Eventually, this module should also implement the Linear Exponential Quadratic
Filters as they were exposed by Peter Whittle (1996)

Note: Inspiration for the structure of this module was taken from the `pykalman` 
module. 


Description 
-----------
Solve the following system of equation using a minimax strategy
    \begin{align}
    x_{k+1} &= F_k*x_k + c_k + nu_k \\
    z_k &= H_k*x_k + d_k + eta_k \\
    y_k &= L_k*x_k
    \end{align}    
    where '*' represents the matrix product $nu_k$ and $eta_k$ are noise terms 
    (of possibly unknown densities), and our goal is to estimate $y_k$.
 
The cost function to solve the problem above is given as:
\[ 
    J_1 = \frac{\sum_{k=0}^{Nobs-1} || z_k - \hat{z}_k||^2_{R_k}}{||x_0 
        - \hat{x}_0||^2_{P_0^{-1}} 
        + \sum_{k=0}^{NObs-1}( ||nu_k||^2_{Q_k^{-1}} + ||eta_k||^2_{S_k^{-1}})} 
\]
The cost function can be made to be less than $\frac{1}{2\gamma}$ 
(a user-specified bound) with a $H_\infty$ or minimax filter

Note: notations are the similar to the ones in my PhD dissertation. 
    Further information on $H_\infty$ filters can be found in 
    "Optimal State Estimation" by Dan Simon, 2006.
    
@author: Guillaume Weisang, Ph.D.
"""

import warnings

import numpy as np
from scipy import linalg

from .utils import array1d, array2d, get_params, preprocess_arguments


class HinfFilter(object):
    """Implements the Minimax filter

    """

    def __init__(self, transition_matrices=None, observation_matrices=None,
                 transition_covariance=None, observation_covariance=None,
                 transition_offsets=None, observation_offsets=None,
                 initial_state_mean=None, initial_state_covariance=None,
                 projection_matrices=None, projection_precision_matrices=None, minimax_bound=1,
                 n_dim_state=None, n_dim_obs=None, n_dim_projs=None):
        """Initialize Minimax Filter's instance

        Check the size of the filter's constituents (parameters) are consistent by determining the size of the
        state space, the size of the observations, and the size of the projections.
        """
        # determine size of state space
        n_dim_state = self._determine_dimensionality(
            [(transition_matrices, array2d, -2),
             (transition_offsets, array1d, -1),
             (transition_covariance, array2d, -2),
             (initial_state_mean, array1d, -1),
             (initial_state_covariance, array2d, -2),
             (observation_matrices, array2d, -1)],
            n_dim_state
        )
        n_dim_obs = self._determine_dimensionality(
            [(observation_matrices, array2d, -2),
             (observation_offsets, array1d, -1),
             (observation_covariance, array2d, -2)],
            n_dim_obs
        )
        n_dim_projs = self._determine_dimensionality(
            [(projection_matrices, array2d, -2),
             (projection_precision_matrices, array2d, -2)],
            n_dim_projs
        )

        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offsets = transition_offsets
        self.observation_offsets = observation_offsets
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.projection_matrices = projection_matrices
        self.projection_precision_matrices = projection_precision_matrices
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self.n_dim_projs = n_dim_projs
        self.minimax_bound = minimax_bound

    # todo: Do I need the method _initialize_parameters?
    def _initialize_parameters(self):
        """Retrieve parameters if they exist, else replace with defaults"""
        n_dim_state, n_dim_obs, n_dim_projs = self.n_dim_state, self.n_dim_obs, self.n_dim_projs

        arguments = get_params(self)
        defaults = {
            'transition_matrices': np.eye(n_dim_state),
            'transition_offsets': np.zeros(n_dim_state),
            'transition_covariance': np.eye(n_dim_state),
            'observation_matrices': np.eye(n_dim_obs, n_dim_state),
            'observation_offsets': np.zeros(n_dim_obs),
            'observation_covariance': np.eye(n_dim_obs),
            'initial_state_mean': np.zeros(n_dim_state),
            'initial_state_covariance': np.eye(n_dim_state),
            'projection_matrices': np.eye(n_dim_state),
            'projection_precision_matrices': np.eye(n_dim_state),
            'minimax_bound': 1
        }
        converters = {
            'transition_matrices': array2d,
            'transition_offsets': array1d,
            'transition_covariance': array2d,
            'observation_matrices': array2d,
            'observation_offsets': array1d,
            'observation_covariance': array2d,
            'initial_state_mean': array1d,
            'initial_state_covariance': array2d,
            'projection_matrices': array2d,
            'projection_precision_matrices': array2d,
            'n_dim_state': int,
            'n_dim_obs': int,
            'n_dim_projs': int,
            'minimax_bound': float
        }

        parameters = preprocess_arguments([arguments, defaults], converters)

        return (
            parameters['transition_matrices'],
            parameters['transition_offsets'],
            parameters['transition_covariance'],
            parameters['observation_matrices'],
            parameters['observation_offsets'],
            parameters['observation_covariance'],
            parameters['initial_state_mean'],
            parameters['initial_state_covariance'],
            parameters['projection_matrices'],
            parameters['projection_precision_matrices'],
            parameters['minimax_bound']
        )

    def filter(self, Y):
        r"""Apply the Minimax filter
        
        Apply the Minimax or :math:`H_\\infty` filter to the hidden state 
        estimate at time :math:`t` for :math:`t = [0...n_{\\text{timesteps}}]`
        given observations up to and including time `t`.  Observations are 
        assumed to correspond to times :math:`[0...n_{\\text{timesteps}}]`
        
        Parameters
        ----------
        Y : [n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `Y` is
            a masked array and any of `Y[t]` is masked, then `Y[t]` will be
            treated as a missing observation.

        Returns
        -------
        filtered_projections_estimates : [n_timesteps, n_dim_projs] array
            the minimax estimate of the projections tracked for times [0...n_timesteps]
        filtered_state_estimates : [n_timesteps, n_dim_state]
            minimax estimates of hidden states for times [0...n_timesteps]
            given observations up to and including the current time step
        filtered_state_pseudocovariances : [n_timesteps, n_dim_state, n_dim_state] \
        array
            pseudo-covariance matrix of hidden states for times
            [0...n_timesteps] given observations up to and including the
            current time step
        corrected_state_estimates: [n_timesteps, n_dimstate] array
            the minimax estimates corrected for the observation at time t for time [0..n_timesteps] \
            (an intermediate calculation step, usually of little interest except for debugging)
        minimax_gains: [n_timesteps, n_dimstates, n_dim_obs] array
            the minimax gains for times [1...n_timesteps]
        """

        Z = self._parse_observations(Y)

        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance, projection_matrices,
         projection_precision_matrices, minimax_bound) = (
            self._initialize_parameters()
        )

        (corrected_state_means, filtered_projections_estimates, minimax_gains, filtered_state_means,
         filtered_state_covariances, minimax_conds) = (
            self._filter(
                transition_matrices, observation_matrices,
                transition_covariance, observation_covariance,
                transition_offsets, observation_offsets, projection_matrices, projection_precision_matrices,
                initial_state_mean, initial_state_covariance,
                Z, minimax_bound
            )
        )
        return (filtered_projections_estimates, filtered_state_means, filtered_state_covariances,
                corrected_state_means, minimax_gains)

    @classmethod
    def _filter(cls, transition_matrices, observation_matrices, transition_covariance,
                observation_covariance, transition_offsets, observation_offsets,
                projection_matrices, projection_precision_matrices, initial_state_mean,
                initial_state_covariance, observations, minimax_bound):
        """Runs the Minimax Filter iterations

        Calculate minimax estimates of hidden states given observations up
        to and including the current time step.

        Parameters
        ----------
        transition_matrices : [n_timesteps-1,n_dim_state,n_dim_state] or
        [n_dim_state,n_dim_state] array-like
            state transition matrices
        observation_matrices : [n_timesteps, n_dim_obs, n_dim_state] or [n_dim_obs, \
        n_dim_state] array-like
            observation matrix
        transition_covariance : [n_timesteps-1,n_dim_state,n_dim_state] or
        [n_dim_state,n_dim_state] array-like
            state transition covariance matrix
        observation_covariance : [n_timesteps, n_dim_obs, n_dim_obs] or [n_dim_obs,
        n_dim_obs] array-like
            observation covariance matrix
        transition_offsets : [n_timesteps-1, n_dim_state] or [n_dim_state] \
        array-like
            state offset
        observation_offsets : [n_timesteps, n_dim_obs] or [n_dim_obs] array-like
            observations for times [0...n_timesteps-1]
        initial_state_mean : [n_dim_state] array-like
            mean of initial state distribution
        initial_state_covariance : [n_dim_state, n_dim_state] array-like
            covariance of initial state distribution
        observations : [n_timesteps, n_dim_obs] array
            observations from times [0...n_timesteps-1].

        Returns
        -------
        corrected_state_means : [n_timesteps-1, n_dim_state] array
            `corrected_state_means[t]` = corrected mean of hidden state at time t given
            observations from times [0...t]
        filtered_state_means : [n_timesteps, n_dim_state] array
            `filtered_state_means[t]` = Minimax estimate of hidden state at time t given
            observations from times [0...t-1]
        filtered_state_covariances : [n_timesteps, n_dim_state] array
            `filtered_state_covariances[t]` = Pseudo-covariance of hidden state at time t
            given observations from times [0...t]
        minimax_gains : [n_timesteps, n_dim_state] array
            `minimax_gains[t]` = Minimax gain matrix for time t
        minimax_bounds : [n_timesteps] array
            `minimax_bounds[t]` = Minimax upper bound for the minimax cost
        minimax_conds: [n_timesteps] array
            `minimax_conds[t]` = True if existence condition of minimax solution is not met
        """

        n_timesteps = observations.shape[0]
        n_dim_state = len(initial_state_mean)
        n_dim_obs = observations.shape[1]
        n_dim_projs = projection_matrices.shape[0]

        filtered_state_means = np.zeros((n_timesteps, n_dim_state))
        corrected_state_means = np.zeros((n_timesteps, n_dim_state))
        filtered_state_covariances = np.zeros(
            (n_timesteps, n_dim_state, n_dim_state)
        )
        minimax_gains = np.zeros((n_timesteps, n_dim_state, n_dim_obs))
        filtered_projections_estimates = np.zeros((n_timesteps, n_dim_projs))
        minimax_conds = np.zeros((n_timesteps, 1))

        # Running the minimax filtering algorithm
        for t in range(n_timesteps):
            if t == 0:
                filtered_state_means[t] = initial_state_mean
                filtered_state_covariances[t] = initial_state_covariance
                filtered_projections_estimates[t] = np.dot(projection_matrices[t], initial_state_mean)
            else:
                transition_matrix = cls._last_dims(transition_matrices, t - 1)
                transition_covariance = cls._last_dims(transition_covariance, t - 1)
                transition_offset = cls._last_dims(transition_offsets, t - 1, ndims=1)
                projection_matrix = cls._last_dims(projection_matrices, t)
                projection_precision = cls._last_dims(projection_precision_matrices, t)
                observation_matrix = cls._last_dims(observation_matrices, t)
                observation_covariance = cls._last_dims(observation_covariance, t)
                observation_offset = cls._last_dims(observation_offsets, t, ndims=1)

                corrected_state_means[t - 1], filtered_state_means[t], filtered_state_covariances[t], \
                filtered_projections_estimates[t], minimax_gains[t], minimax_conds[t] = (
                    cls._filter_step(
                        transition_matrix,
                        transition_covariance,
                        transition_offset,
                        projection_matrix,
                        projection_precision,
                        filtered_state_means[t - 1],
                        filtered_state_covariances[t - 1],
                        observation_matrix,
                        observation_covariance,
                        observation_offset,
                        observations[t],
                        minimax_bound
                    )
                )

        return (corrected_state_means, filtered_projections_estimates, minimax_gains, filtered_state_means,
                filtered_state_covariances, minimax_conds)

    @staticmethod
    def _filter_step(transition_matrix, transition_covariance, transition_offset, projection_matrix,
                     projection_precision, filtered_state_mean, filtered_state_covariance, observation_matrix,
                     observation_covariance, observation_offset, observation, minimax_bound):
        """Apply one step of the Minimax filter

        Calculate the one-step recursion of the Minimax filter satisfying the minimax bound.

        Parameters
        ----------
        :rtype : tuple of numpy.ndarrays
        :param transition_matrix:
        :param transition_covariance:
        :param transition_offset:
        :param projection_matrix:
        :param projection_precision:
        :param filtered_state_mean:
        :param filtered_state_covariance:
        :param observation_matrix:
        :param observation_covariance:
        :param observation_offset:
        :param observation:
        :param minimax_bound:
        :return:
        """
        minimax_cond = False
        projected_precision = np.dot(np.dot(np.transpose(projection_matrix), projection_precision), projection_matrix)
        # Inverting observation covariance matrix
        try:
            inv_observation_covariance = linalg.inv(observation_covariance)
        except linalg.LinAlgError:
            warn_str = 'Inversion of observation covariance matrix failed. '
            + 'Moving to Moore-Penrose pseudo-inverse'
            warnings.warn(warn_str)
            try:
                inv_observation_covariance = linalg.pinv(observation_covariance)
            except linalg.LinAlgError:
                warn_str = 'SVD Calculation did not converge.'
                warnings.warn(warn_str)
                raise ValueError(warn_str) from linalg.LinAlgError

        # Inverting previous pseudo state covariance
        try:
            inv_state_covariance = linalg.inv(filtered_state_covariance)
        except linalg.LinAlgError:
            warn_str = 'Inversion of pseudo state covariance failed. Moving to Moore-Penrose pseudo-inverse'
            warnings.warn(warn_str)
            try:
                inv_state_covariance = linalg.pinv(filtered_state_covariance)
            except linalg.LinAlgError:
                warn_str = 'SVD Calculation did not converge.'
                warnings.warn(warn_str)
                raise ValueError(warn_str) from linalg.LinAlgError

        # Testing the condition of existence of a solution
        filtered_state_covariance_tilde = inv_state_covariance - minimax_bound * projected_precision + \
                                          np.dot(np.dot(observation_matrix.T, inv_observation_covariance),
                                                 observation_matrix
                                                 )
        try:
            filtered_state_covariance_tilde_inv = linalg.inv(filtered_state_covariance_tilde)
        except linalg.LinAlgError:
            minimax_cond = True
            warn_str = (
            'Condition of existence of minimax solution failed for minimax bound = {0}. ' ).format(minimax_bound)
            warnings.warn(warn_str)
            # try:
            #     filtered_state_covariance_tilde = linalg.pinv(inv_state_covariance -
            #                                                   minimax_bound * projected_precision +
            #                                                   np.dot(np.dot(observation_matrix.T,
            #                                                                 inv_observation_covariance),
            #                                                          observation_matrix
            #                                                          )
            #                                                   )
            # except linalg.LinAlgError:
            #     warn_str = 'SVD Calculation did not converge.'
            #     warnings.warn(warn_str)
            #     raise ValueError(warn_str) from linalg.LinAlgError
            # finally:
            #     minimax_cond = True

        # Calculating the Minimax gain
        minimax_gain = np.dot(np.dot(filtered_state_covariance_tilde, observation_matrix.T),
                              inv_observation_covariance)

        # Calculating the innovation
        minimax_innovation = observation - np.dot(observation_matrix,
                                                  filtered_state_mean + transition_offset) - observation_offset

        # Calculating the corrected state mean
        corrected_state_mean = filtered_state_mean + np.dot(minimax_gain, minimax_innovation)

        # Calculating the new filtered_state_mean
        filtered_state_mean = np.dot(transition_matrix, corrected_state_mean)

        # Calculating the new filtered_state_pseudo_covariance
        filtered_state_covariance = np.dot(np.dot(transition_matrix, filtered_state_covariance_tilde),
                                           transition_matrix.T) + transition_covariance
        # Calculating the new projection_estimate
        projection_estimate = np.dot(projection_matrix, filtered_state_mean)

        return (corrected_state_mean, filtered_state_mean, filtered_state_covariance,
                projection_estimate, minimax_gain, minimax_cond)

    @staticmethod
    def _parse_observations(obs):
        """Safely convert observations to their expected format"""
        obs = np.ma.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        return obs

    @staticmethod
    def _determine_dimensionality(variables, default):
        """Derive the dimensionality of the state space

        Parameters
        ----------
        variables : list of ({None, array}, conversion function, index)
            variables, functions to convert them to arrays, and indices in those
            arrays to derive dimensionality from.
        default : {None, int}
            default dimensionality to return if variables is empty

        Returns
        -------
        dim : int
            dimensionality of state space as derived from variables or default.
        """
        # gather possible values based on the variables
        candidates = []
        for (v, converter, idx) in variables:
            if v is not None:
                v = converter(v)
                candidates.append(v.shape[idx])

        # also use the manually specified default
        if default is not None:
            candidates.append(default)

        # ensure consistency of all derived values
        if len(candidates) == 0:
            return 1
        else:
            if not np.all(np.array(candidates) == candidates[0]):
                raise ValueError(
                    "The shape of all " +
                    "parameters is not consistent.  " +
                    "Please re-check their values."
                )
            return candidates[0]

    @staticmethod
    def _last_dims(arr, t, ndims=2):
        """Extract the final dimensions of `arr`
    
        Extract the final `ndim` dimensions at index `t` if `arr` has >= `ndim` + 1
        dimensions, otherwise return `arr`.
    
        Parameters
        ----------
        arr : array with at least dimension `ndims`
        t : int
            index to use for the `ndims` + 1th dimension
        ndims : int, optional
            number of dimensions in the array desired
    
        Returns
        -------
        Y : array with dimension `ndims`
            the final `ndims` dimensions indexed by `t`
        """
        arr = np.asarray(arr)
        if len(arr.shape) == ndims + 1:
            return arr[t]
        elif len(arr.shape) == ndims:
            return arr
        else:
            raise ValueError(("Array only has %d dimensions when %d" +
                              " or more are required") % (len(arr.shape), ndims))
