'''
Defines the frequentist model (where model = data model + statistical model).
'''

import numpy as np
from brick import AZR
from priors import cmes

azr = AZR('model.azr')
azr.ext_capture_file = 'output/intEC.dat'
azr.root_directory = '/tmp/'

def norm_factor_penalty(theta):
    norm_factors = theta[azr.config.n1:]
    return np.sum([((n - 1)/cme)**2 for (n, cme) in zip(norm_factors, cmes)])


def chi_squared(theta):
    scatter, capture_ground, capture_excited, capture_total = azr.predict(theta, dress_up=False)

    # The branching ratio prediction requires dividing the excited state cross
    # ection by the ground state.
    branching_ratio = capture_excited[:, 3] / capture_ground[:, 3]
    mu = np.hstack((
        scatter[:, 3],
        branching_ratio,
        capture_total[:, 3]
    ))

    # The branching ratio _data_ is input as data in the files, so we don't have
    # to compute the ratio.
    y = np.hstack((
        scatter[:, 5],
        capture_ground[:, 5],
        capture_total[:, 6]
    ))
    dy = np.hstack((
        scatter[:, 6],
        capture_ground[:, 6],
        capture_total[:, 6]
    ))

    return np.sum(((y - mu)/dy)**2) + norm_factor_penalty(theta)
