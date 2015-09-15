===============================================================================
                Minimax Inference for Linear Systems
===============================================================================

This module implements the minimax filter as described by [Simon2006]_.
Eventually, this module should very soon also implement the Linear Exponential Quadratic
Filters as they were exposed by [Whittle1996]_.



-----------
Description
-----------
This module implements filters that solve the following system of equation using a minimax strategy

.. math::

    \mathbf{x}_{k+1} &= \mathbf{F}_k \mathbf{x}_k + \mathbf{c}_k + \boldsymbol\nu_k \\
    \mathbf{z}_k &= \mathbf{H}_k \mathbf{x}_k + \mathbf{d}_k + \boldsymbol\eta_k \\
    \mathbf{y}_k &= \mathbf{L}_k \mathbf{x}_k

where :math:`\nu_k` and :math:`\eta_k` are noise terms (of possibly unknown densities), and our goal is to estimate :math:`y_k`.

The cost function to solve the problem above is given as:

.. math::

    J_1 = \frac{\sum_{k=0}^{N-1} || \mathbf{z}_k - \hat{\mathbf{z}}_k||^2_{\mathbf{R}_k}}{||x_0
        - \hat{x}_0||^2_{\mathbf{P}_0^{-1}}
        + \sum_{k=0}^{N-1}( ||\nu_k||^2_{\mathbf{Q}_k^{-1}} + ||\eta_k||^2_{\mathbf{S}_k^{-1}})}

The cost function can be made to be less than :math:`\frac{1}{2\gamma}`
(a user-specified bound) with a :math:`H_\infty` or minimax filter.


------------
References
------------

.. [Simon2006] Simon, Dan. 2006. Optimal State Estimation. Wiley-Interscience.
.. [Whittle1996] Whittle, Peter. 1996. Optimal Control: Basics and Beyond. 1st ed. Series in Systems and Optimization. Wiley-Interscience.


.. note::
    Notations are the similar to the ones in my PhD dissertation, and correspond mostly to notations used by [Simon2006]_

.. note:: Inspiration for the structure of this module was taken from the ``pykalman`` module.
