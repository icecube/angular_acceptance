# version db2550e747331de00dc9879331ae5681
from __future__ import division, print_function
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline
from numpy.polynomial import polynomial

# these values are obtained from a fit (explained in https://github.com/philippeller/angular_acceptance/blob/master/Angular_acceptance.ipynb)
support_x = np.array([-1.  , -0.6 ,  0.2 ,  0.6 ,  0.75,  0.9 ,  1.  ])
components = np.array([[ 0.        , -0.04173206, -0.08273521, -0.09952227,  0.04633073,
         0.4499595 ,  0.88141849],
       [ 0.        , -0.19142412, -0.17404567,  0.46835844,  0.73826994,
         0.36108428, -0.1956551 ]])
mean = np.array([0.        , 0.08878613, 0.43311141, 0.62266136, 0.6649289 ,
       0.61862511, 0.54641979])
n_components = 2

def ang(params, values):
    """
    New angular acceptance function
    
    params : list / array
        the parameters p0, p1, ...
    values : float, list, array
        the eta values to compute the angular acceptance for in (-1, 1)
    """
    # sanity check
    assert np.all(np.logical_and(np.greater_equal(values, -1), np.less_equal(values, 1))), 'values must be in range -1, 1'
    p = np.zeros(n_components)
    p[:len(params)] = params
    # inverse PCA transform
    transformed_params = np.dot(p, components) + mean
    # construct spline
    f = make_interp_spline(support_x, transformed_params, bc_type=([(1, 0.)], [(1, 0.)]))
    # make sure we're positive everywhere
    positive_f = lambda x : np.clip(f(x), 0., None)
    # normalize
    norm = quad(positive_f, -1, 1)
    out = positive_f(values)
    #normalize....why 0.68?
    out *= 0.68 / norm[0]
    return out
    
def angsens_poly(params):
    """
    Return standard IceCube format
    
    params : list/array
        the parameters p0, p1, ... 
    """
    
    x = np.linspace(-1,1,101)
    sampled = ang(params, x)
    coeffs = np.zeros(12)
    coeffs[0] = np.max(sampled)
    coeffs[1:] = polynomial.polyfit(x, sampled, 10)
    return coeffs
    
if __name__=='__main__':
    from sys import argv
    params = [0.]*n_components
    if len(argv) > 0:
        params[:len(argv)-1] = [float(a) for a in argv[1:]]
    print('\n'.join([str(a) for a in angsens_poly(params)]))
