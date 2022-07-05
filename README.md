# Unified Angular Acceptance (aka "Hole Ice") Model

This repo describes the unified model in the jupyter notebook

Two small python scripts are also provided:

* `unified_angular_acceptance.py` : the __new__ model with two parameters `p0` and `p1`. Script generates IceTray compatible `as` polynomials, use with the parameter values as arguments, e.g. `./unified_angular_acceptance.py 0.2 -0.1 > as.new`
* `make_dima_msu_polynomial.py` : the __old__ model with the "Dima" + "MSU forward" parameters, generates IceTray compatible `as` polynomials, use with the parameter values as arguments, e.g. `./make_dima_msu_polynomial.py 0.3 -1 > as.old`

Contact `peller@icecube.wisc.edu` for questions, or ping me on slack!
