'''
This is from https://github.com/scipy/scipy/blob/v1.2.1/scipy/interpolate/_bsplines.py


Since we're not able to have modern scipy version in iceTray
'''
from __future__ import division, print_function, absolute_import

import operator

import numpy as np
from scipy._lib.six import string_types
from scipy.linalg import (get_lapack_funcs, LinAlgError,
                          cholesky_banded, cho_solve_banded)
from scipy.interpolate import _bspl


#################################
#  Interpolating spline helpers #
#################################

def _not_a_knot(x, k):
    """Given data x, construct the knot vector w/ not-a-knot BC.
    cf de Boor, XIII(12)."""
    x = np.asarray(x)
    if k % 2 != 1:
        raise ValueError("Odd degree for now only. Got %s." % k)

    m = (k - 1) // 2
    t = x[m+1:-m-1]
    t = np.r_[(x[0],)*(k+1), t, (x[-1],)*(k+1)]
    return t


def _augknt(x, k):
    """Construct a knot vector appropriate for the order-k interpolation."""
    return np.r_[(x[0],)*k, x, (x[-1],)*k]


def _convert_string_aliases(deriv, target_shape):
    if isinstance(deriv, string_types):
        if deriv == "clamped":
            deriv = [(1, np.zeros(target_shape))]
        elif deriv == "natural":
            deriv = [(2, np.zeros(target_shape))]
        else:
            raise ValueError("Unknown boundary condition : %s" % deriv)
    return deriv


def _process_deriv_spec(deriv):
    if deriv is not None:
        try:
            ords, vals = zip(*deriv)
        except TypeError:
            msg = ("Derivatives, `bc_type`, should be specified as a pair of "
                   "iterables of pairs of (order, value).")
            raise ValueError(msg)
    else:
        ords, vals = [], []
    return np.atleast_1d(ords, vals)


def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0,
                       check_finite=True):
    """Compute the (coefficients of) interpolating B-spline.
    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n, ...)
        Ordinates.
    k : int, optional
        B-spline degree. Default is cubic, k=3.
    t : array_like, shape (nt + k + 1,), optional.
        Knots.
        The number of knots needs to agree with the number of datapoints and
        the number of derivatives at the edges. Specifically, ``nt - n`` must
        equal ``len(deriv_l) + len(deriv_r)``.
    bc_type : 2-tuple or None
        Boundary conditions.
        Default is None, which means choosing the boundary conditions
        automatically. Otherwise, it must be a length-two tuple where the first
        element sets the boundary conditions at ``x[0]`` and the second
        element sets the boundary conditions at ``x[-1]``. Each of these must
        be an iterable of pairs ``(order, value)`` which gives the values of
        derivatives of specified orders at the given edge of the interpolation
        interval.
        Alternatively, the following string aliases are recognized:
        * ``"clamped"``: The first derivatives at the ends are zero. This is
           equivalent to ``bc_type=([(1, 0.0)], [(1, 0.0)])``.
        * ``"natural"``: The second derivatives at ends are zero. This is
          equivalent to ``bc_type=([(2, 0.0)], [(2, 0.0)])``.
        * ``"not-a-knot"`` (default): The first and second segments are the same
          polynomial. This is equivalent to having ``bc_type=None``.
    axis : int, optional
        Interpolation axis. Default is 0.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.
    Returns
    -------
    b : a BSpline object of the degree ``k`` and with knots ``t``.
    Examples
    --------
    Use cubic interpolation on Chebyshev nodes:
    >>> def cheb_nodes(N):
    ...     jj = 2.*np.arange(N) + 1
    ...     x = np.cos(np.pi * jj / 2 / N)[::-1]
    ...     return x
    >>> x = cheb_nodes(20)
    >>> y = np.sqrt(1 - x**2)
    >>> from scipy.interpolate import BSpline, make_interp_spline
    >>> b = make_interp_spline(x, y)
    >>> np.allclose(b(x), y)
    True
    Note that the default is a cubic spline with a not-a-knot boundary condition
    >>> b.k
    3
    Here we use a 'natural' spline, with zero 2nd derivatives at edges:
    >>> l, r = [(2, 0.0)], [(2, 0.0)]
    >>> b_n = make_interp_spline(x, y, bc_type=(l, r))  # or, bc_type="natural"
    >>> np.allclose(b_n(x), y)
    True
    >>> x0, x1 = x[0], x[-1]
    >>> np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])
    True
    Interpolation of parametric curves is also supported. As an example, we
    compute a discretization of a snail curve in polar coordinates
    >>> phi = np.linspace(0, 2.*np.pi, 40)
    >>> r = 0.3 + np.cos(phi)
    >>> x, y = r*np.cos(phi), r*np.sin(phi)  # convert to Cartesian coordinates
    Build an interpolating curve, parameterizing it by the angle
    >>> from scipy.interpolate import make_interp_spline
    >>> spl = make_interp_spline(phi, np.c_[x, y])
    Evaluate the interpolant on a finer grid (note that we transpose the result
    to unpack it into a pair of x- and y-arrays)
    >>> phi_new = np.linspace(0, 2.*np.pi, 100)
    >>> x_new, y_new = spl(phi_new).T
    Plot the result
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o')
    >>> plt.plot(x_new, y_new, '-')
    >>> plt.show()
    See Also
    --------
    BSpline : base class representing the B-spline objects
    CubicSpline : a cubic spline in the polynomial basis
    make_lsq_spline : a similar factory function for spline fitting
    UnivariateSpline : a wrapper over FITPACK spline fitting routines
    splrep : a wrapper over FITPACK spline fitting routines
    """
    # convert string aliases for the boundary conditions
    if bc_type is None or bc_type == 'not-a-knot':
        deriv_l, deriv_r = None, None
    elif isinstance(bc_type, string_types):
        deriv_l, deriv_r = bc_type, bc_type
    else:
        try:
            deriv_l, deriv_r = bc_type
        except TypeError:
            raise ValueError("Unknown boundary condition: %s" % bc_type)

    y = np.asarray(y)

    if not -y.ndim <= axis < y.ndim:
        raise ValueError("axis {} is out of bounds".format(axis))
    if axis < 0:
        axis += y.ndim

    # special-case k=0 right away
    if k == 0:
        if any(_ is not None for _ in (t, deriv_l, deriv_r)):
            raise ValueError("Too much info for k=0: t and bc_type can only "
                             "be None.")
        x = _as_float_array(x, check_finite)
        t = np.r_[x, x[-1]]
        c = np.asarray(y)
        c = np.rollaxis(c, axis)
        c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)

    # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
    if k == 1 and t is None:
        if not (deriv_l is None and deriv_r is None):
            raise ValueError("Too much info for k=1: bc_type can only be None.")
        x = _as_float_array(x, check_finite)
        t = np.r_[x[0], x, x[-1]]
        c = np.asarray(y)
        c = np.rollaxis(c, axis)
        c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)

    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    k = operator.index(k)

    # come up with a sensible knot vector, if needed
    if t is None:
        if deriv_l is None and deriv_r is None:
            if k == 2:
                # OK, it's a bit ad hoc: Greville sites + omit
                # 2nd and 2nd-to-last points, a la not-a-knot
                t = (x[1:] + x[:-1]) / 2.
                t = np.r_[(x[0],)*(k+1),
                           t[1:-1],
                           (x[-1],)*(k+1)]
            else:
                t = _not_a_knot(x, k)
        else:
            t = _augknt(x, k)

    t = _as_float_array(t, check_finite)

    y = np.rollaxis(y, axis)    # now internally interp axis is zero

    if x.ndim != 1 or np.any(x[1:] <= x[:-1]):
        raise ValueError("Expect x to be a 1-D sorted array_like.")
    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t.ndim != 1 or np.any(t[1:] < t[:-1]):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    if x.size != y.shape[0]:
        raise ValueError('x and y are incompatible.')
    if t.size < x.size + k + 1:
        raise ValueError('Got %d knots, need at least %d.' %
                         (t.size, x.size + k + 1))
    if (x[0] < t[k]) or (x[-1] > t[-k]):
        raise ValueError('Out of bounds w/ x = %s.' % x)

    # Here : deriv_l, r = [(nu, value), ...]
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]

    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]

    # have `n` conditions for `nt` coefficients; need nt-n derivatives
    n = x.size
    nt = t.size - k - 1

    if nt - n != nleft + nright:
        raise ValueError("The number of derivatives at boundaries does not "
                         "match: expected %s, got %s+%s" % (nt-n, nleft, nright))

    # set up the LHS: the collocation matrix + derivatives at boundaries
    kl = ku = k
    ab = np.zeros((2*kl + ku + 1, nt), dtype=np.float_, order='F')
    _bspl._colloc(x, t, k, ab, offset=nleft)
    if nleft > 0:
        _bspl._handle_lhs_derivatives(t, k, x[0], ab, kl, ku, deriv_l_ords)
    if nright > 0:
        _bspl._handle_lhs_derivatives(t, k, x[-1], ab, kl, ku, deriv_r_ords,
                                offset=nt-nright)

    # set up the RHS: values to interpolate (+ derivative values, if any)
    extradim = prod(y.shape[1:])
    rhs = np.empty((nt, extradim), dtype=y.dtype)
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)

    # solve Ab @ x = rhs; this is the relevant part of linalg.solve_banded
    if check_finite:
        ab, rhs = map(np.asarray_chkfinite, (ab, rhs))
    gbsv, = get_lapack_funcs(('gbsv',), (ab, rhs))
    lu, piv, c, info = gbsv(kl, ku, ab, rhs,
            overwrite_ab=True, overwrite_b=True)

    if info > 0:
        raise LinAlgError("Collocation matix is singular.")
    elif info < 0:
        raise ValueError('illegal value in %d-th argument of internal gbsv' % -info)

    c = np.ascontiguousarray(c.reshape((nt,) + y.shape[1:]))
    return BSpline.construct_fast(t, c, k, axis=axis)


def make_lsq_spline(x, y, t, k=3, w=None, axis=0, check_finite=True):
    r"""Compute the (coefficients of) an LSQ B-spline.
    The result is a linear combination
    .. math::
            S(x) = \sum_j c_j B_j(x; t)
    of the B-spline basis elements, :math:`B_j(x; t)`, which minimizes
    .. math::
        \sum_{j} \left( w_j \times (S(x_j) - y_j) \right)^2
    Parameters
    ----------
    x : array_like, shape (m,)
        Abscissas.
    y : array_like, shape (m, ...)
        Ordinates.
    t : array_like, shape (n + k + 1,).
        Knots.
        Knots and data points must satisfy Schoenberg-Whitney conditions.
    k : int, optional
        B-spline degree. Default is cubic, k=3.
    w : array_like, shape (n,), optional
        Weights for spline fitting. Must be positive. If ``None``,
        then weights are all equal.
        Default is ``None``.
    axis : int, optional
        Interpolation axis. Default is zero.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.
    Returns
    -------
    b : a BSpline object of the degree `k` with knots `t`.
    Notes
    -----
    The number of data points must be larger than the spline degree `k`.
    Knots `t` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.
    Examples
    --------
    Generate some noisy data:
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * np.random.randn(50)
    Now fit a smoothing cubic spline with a pre-defined internal knots.
    Here we make the knot vector (k+1)-regular by adding boundary knots:
    >>> from scipy.interpolate import make_lsq_spline, BSpline
    >>> t = [-1, 0, 1]
    >>> k = 3
    >>> t = np.r_[(x[0],)*(k+1),
    ...           t,
    ...           (x[-1],)*(k+1)]
    >>> spl = make_lsq_spline(x, y, t, k)
    For comparison, we also construct an interpolating spline for the same
    set of data:
    >>> from scipy.interpolate import make_interp_spline
    >>> spl_i = make_interp_spline(x, y)
    Plot both:
    >>> import matplotlib.pyplot as plt
    >>> xs = np.linspace(-3, 3, 100)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ spline')
    >>> plt.plot(xs, spl_i(xs), 'b-', lw=3, alpha=0.7, label='interp spline')
    >>> plt.legend(loc='best')
    >>> plt.show()
    **NaN handling**: If the input arrays contain ``nan`` values, the result is
    not useful since the underlying spline fitting routines cannot deal with
    ``nan``. A workaround is to use zero weights for not-a-number data points:
    >>> y[8] = np.nan
    >>> w = np.isnan(y)
    >>> y[w] = 0.
    >>> tck = make_lsq_spline(x, y, t, w=~w)
    Notice the need to replace a ``nan`` by a numerical value (precise value
    does not matter as long as the corresponding weight is zero.)
    See Also
    --------
    BSpline : base class representing the B-spline objects
    make_interp_spline : a similar factory function for interpolating splines
    LSQUnivariateSpline : a FITPACK-based spline fitting routine
    splrep : a FITPACK-based fitting routine
    """
    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    t = _as_float_array(t, check_finite)
    if w is not None:
        w = _as_float_array(w, check_finite)
    else:
        w = np.ones_like(x)
    k = operator.index(k)

    if not -y.ndim <= axis < y.ndim:
        raise ValueError("axis {} is out of bounds".format(axis))
    if axis < 0:
        axis += y.ndim

    y = np.rollaxis(y, axis)    # now internally interp axis is zero

    if x.ndim != 1 or np.any(x[1:] - x[:-1] <= 0):
        raise ValueError("Expect x to be a 1-D sorted array_like.")
    if x.shape[0] < k+1:
        raise ValueError("Need more x points.")
    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t.ndim != 1 or np.any(t[1:] - t[:-1] < 0):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    if x.size != y.shape[0]:
        raise ValueError('x & y are incompatible.')
    if k > 0 and np.any((x < t[k]) | (x > t[-k])):
        raise ValueError('Out of bounds w/ x = %s.' % x)
    if x.size != w.size:
        raise ValueError('Incompatible weights.')

    # number of coefficients
    n = t.size - k - 1

    # construct A.T @ A and rhs with A the collocation matrix, and
    # rhs = A.T @ y for solving the LSQ problem  ``A.T @ A @ c = A.T @ y``
    lower = True
    extradim = prod(y.shape[1:])
    ab = np.zeros((k+1, n), dtype=np.float_, order='F')
    rhs = np.zeros((n, extradim), dtype=y.dtype, order='F')
    _bspl._norm_eq_lsq(x, t, k,
                      y.reshape(-1, extradim),
                      w,
                      ab, rhs)
    rhs = rhs.reshape((n,) + y.shape[1:])

    # have observation matrix & rhs, can solve the LSQ problem
    cho_decomp = cholesky_banded(ab, overwrite_ab=True, lower=lower,
                                 check_finite=check_finite)
    c = cho_solve_banded((cho_decomp, lower), rhs, overwrite_b=True,
                         check_finite=check_finite)

    c = np.ascontiguousarray(c)
    return BSpline.construct_fast(t, c, k, axis=axis)
