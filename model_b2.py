'''
Defines the frequentist model (where model = data model + statistical model).
'''

import numpy as np
from brick import AZR
from priors import cmes, priors

azr = AZR('model_2.azr')
azr.ext_capture_file = 'output/intEC.dat'
azr.root_directory = '/tmp/'
# azr.command = '/Applications/AZURE2.app/Contents/MacOS/AZURE2'

dy_no_norm = np.loadtxt('bare_uncertainties_2.txt')

def ln_prior(theta):
    return np.sum([p.logpdf(x) for (p, x) in zip(priors, theta)])


def ln_likelihood(theta):
    capture_ground, capture_excited, capture_total = azr.predict(theta, dress_up=False)

    # The branching ratio prediction requires dividing the excited state cross
    # ection by the ground state.
    branching_ratio = capture_excited[:, 3] / capture_ground[:, 3]
    mu = np.hstack((
        branching_ratio,
        capture_total[:, 3]
    ))

    # The branching ratio _data_ is input as data in the files, so we don't have
    # to compute the ratio.
    y = np.hstack((
        capture_ground[:, 5],
        capture_total[:, 5]
    ))
    dy = np.hstack((
        capture_ground[:, 6],
        capture_total[:, 6]
    ))

    return np.sum(np.log(1/(np.sqrt(2*np.pi)*dy_no_norm)) - ((y - mu)/dy)**2)


def ln_posterior(theta):
    lnpi = ln_prior(theta)
    if lnpi == -np.inf:
        return -np.inf

    lnl = ln_likelihood(theta)
    return lnl + lnpi
