'''
Prior distributions common to all analyses.
'''

import numpy as np
from scipy import stats

def my_truncnorm(mu, sigma, lower, upper):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)


# priors
anc429_prior = stats.uniform(1, 4)
Ga_1hm_prior = stats.uniform(-200e6, 400e6)

Ga_1hp_prior = stats.uniform(0, 100e6)
Gg0_1hp_prior = stats.uniform(0, 10e6)
Gg429_1hp_prior = stats.uniform(-20e3, 40e3)

anc0_prior = stats.uniform(1, 4)
Ga_3hm_prior = stats.uniform(-100e6, 200e6)

Ga_3hp_prior = stats.uniform(0, 1e6)
# Gg0_3hp_prior = stats.uniform(-10e3, 20e3)
# Gg429_3hp_prior = stats.uniform(-3e3, 6e3)

Ga_5hm_prior = stats.uniform(0, 100e6)
Ga_5hp_prior = stats.uniform(0, 100e6)
# Gg0_5hp_prior = stats.uniform(-100e6, 200e6)

# Ex_7hm_prior = stats.uniform(1, 9)
# Ga_7hm_prior = stats.uniform(0, 10e6)
# Gg0_7hm_prior = stats.uniform(0, 1e3)

cme_seattle = 0.03
cme_weizmann = 0.022
cme_luna = 0.032
cme_erna = 0.05
cme_atomki = 0.059
cme_cmam = 0.03

f_seattle_prior = my_truncnorm(1, cme_seattle, 0, 10)
f_weizmann_prior = my_truncnorm(1, cme_weizmann, 0, 10)
f_luna_prior = my_truncnorm(1, cme_luna, 0, 10)
f_erna_prior = my_truncnorm(1, cme_erna, 0, 10)
f_atomki_prior = my_truncnorm(1, cme_atomki, 0, 10)
f_cmam_prior = my_truncnorm(1, cme_cmam, 0, 10)

f_capture_priors = [
    f_seattle_prior,
    f_weizmann_prior,
    f_luna_prior,
    f_erna_prior,
    f_atomki_prior,
    f_cmam_prior
]

# Add 1% (in quadrature) to systematic uncertainty to account for beam-position
# uncertainty.
sonik_syst_1820 = 0.089
sonik_syst_1441 = 0.063
sonik_syst_1196 = 0.077
sonik_syst_873_2 = 0.041
sonik_syst_873_1 = 0.062
sonik_syst_711 = 0.045
sonik_syst_586 = 0.057
sonik_syst_432 = 0.098
sonik_syst_291 = 0.076
sonik_syst_239 = 0.064

sonik_syst = np.array([
    sonik_syst_239,
    sonik_syst_291,
    sonik_syst_432,
    sonik_syst_586,
    sonik_syst_711,
    sonik_syst_873_1,
    sonik_syst_873_2
#     sonik_syst_1196,
#     sonik_syst_1441,
#     sonik_syst_1820
])

cmes = np.hstack((
    cme_seattle,
    cme_weizmann,
    cme_luna,
    cme_erna,
    cme_atomki,
    cme_cmam,
    sonik_syst
))

sonik_syst = np.sqrt(sonik_syst**2 + 0.01**2)
sonik_priors = [my_truncnorm(1, syst, 0, np.inf) for syst in sonik_syst]

priors = [
    anc429_prior,
    Ga_1hm_prior,
    Ga_1hp_prior,
    Gg0_1hp_prior,
    Gg429_1hp_prior,
    anc0_prior,
    Ga_3hm_prior,
    Ga_3hp_prior,
#     Gg0_3hp_prior,
#     Gg429_3hp_prior,
    Ga_5hm_prior,
    Ga_5hp_prior,
#     Gg0_5hp_prior,
    # Ex_7hm_prior,
#     Ga_7hm_prior,
#     Gg0_7hm_prior
] + f_capture_priors + sonik_priors

