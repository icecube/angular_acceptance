from __future__ import division
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.integrate import quad

# these values are obtained from a fit
support_x = np.array([-1.  , -0.6 , -0.25,  0.1 ,  0.5 ,  0.9 ,  1.  ])
components = np.array([[ 0.        ,  0.02185064, -0.02789483,  0.10264894,  0.15271317,
        -0.47946672, -0.85732023],
       [-0.        , -0.36052758, -0.64729019, -0.06885077,  0.56835672,
         0.34050616, -0.08556311]])
mean = np.array([0.        , 0.10226867, 0.24635475, 0.36417989, 0.54721411,
       0.63978273, 0.58457966])

def ang(params, values):
    """
    New angular acceptance function
    
    params : list / array
        the two parametrrs p0 and p1
    values : float, list, array
        the eta values to compute the angular acceptance for in (-1, 1)
    """
    # sanity check
    assert np.all(np.logical_and(np.greater_equal(values, -1), np.less_equal(values, 1))), 'values must be in range -1, 1'
    # inverse PCA transform
    transformed_params = np.dot(params, components) + mean
    # construct spline
    f = make_interp_spline(support_x, transformed_params, bc_type=([(1, 0.)], [(1, 0.)]))
    # make sure we're positive averywhere
    positive_f = lambda x : np.clip(f(x), 0., None)
    # normalize
    norm = quad(positive_f, -1, 1)
    out = positive_f(values)
    #normalize....why 0.68?
    out *= 0.68 / norm[0]
    return out
