#!/usr/bin/env python
from __future__ import division, print_function
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline
from numpy.polynomial import polynomial


'''small module to produce icetray compatible outputs (normalization + 10d polynomial)

Useage example:
---------------
./make_dima_msu_polynomial.py 0.3 -1 > as.mynewmodel
'''

def msu(p, coseta):
    '''
    MSU parameterization 
    https://wiki.icecube.wisc.edu/index.php/MSU_Forward_Hole_Ice
    '''
    f = lambda x : 0.34*(1. + 1.5 * x - x**3/2.) + p[0] * x * (x**2 - 1.)**3 + p[1] * np.exp(10.*(x - 1.2))
    norm = quad(f, -1, 1)
    return f(coseta) * 0.68 / norm[0]
    return f(coseta)
    
def msu_poly(params):
    """
    Return standard IceCube format
    
    params : list/array
        the parameters p0, p1, ... 
    """
    
    x = np.linspace(-1,1,1001)
    sampled = msu(params, x)
    coeffs = np.zeros(12)
    coeffs[0] = np.max(sampled)
    coeffs[1:] = polynomial.polyfit(x, sampled, 10)
    return coeffs
    
if __name__=='__main__':
    from sys import argv
    assert len(argv) == 3
    params = [float(a) for a in argv[1:]]
    print('\n'.join([str(a) for a in msu_poly(params)]))
