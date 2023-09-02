#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import model_bayesian as model
import emcee
from multiprocess import Pool
from scipy import stats
import scienceplots

plt.style.use('science')


# In[2]:


os.environ['OMP_NUM_THREADS'] = '1'


# In[3]:


theta0 = model.azr.config.get_input_values()


# In[4]:


scatter, capture_ground, capture_excited, capture_total = model.azr.predict(theta0, dress_up=False)


# In[5]:


fig, ax = plt.subplots(dpi=200)
fig.patch.set_facecolor('white')

ax.scatter(capture_excited[:, 0], capture_excited[:, 3] / capture_ground[:, 3])
ax.errorbar(capture_excited[:, 0], capture_excited[:, 5], yerr=capture_excited[:, 6], linestyle='', capsize=3)

ax.set_xlabel(r'$E$ (MeV')
ax.set_ylabel('Branching Ratio');


# In[6]:


# np.savetxt('bare_uncertainties.txt', np.vstack((scatter, capture_ground, capture_total))[:, 6])


# In[7]:


model.ln_prior(theta0)


# In[8]:


model.ln_likelihood(theta0)


# In[9]:


model.ln_posterior(theta0)


# In[10]:


nd = model.azr.config.n1 + model.azr.config.n2
nw = 2*nd

theta0 = model.azr.config.get_input_values()
p0 = np.array([[stats.norm(x, np.abs(x)/1000).rvs() for x in theta0] for _ in range(nw)])


# In[11]:


with Pool(8) as pool:
    sampler = emcee.EnsembleSampler(nw, nd, model.ln_posterior, pool=pool,
                                    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
    state = sampler.run_mcmc(p0, 100, thin_by=1, progress=True)


# In[12]:


lnp = sampler.get_log_prob()


# In[13]:


nb = 0
min_lnp = -np.inf

good_walkers = np.where(np.min(lnp[nb:], axis=0) > min_lnp)[0]

fig, ax = plt.subplots(dpi=200)
fig.patch.set_facecolor('white')

ax.plot(lnp[nb:, good_walkers]);


# In[14]:


chain = sampler.get_chain(discard=nb)[:, good_walkers, :]


# In[15]:


chain.shape


# In[16]:


emcee.autocorr.integrated_time(chain)


# In[17]:


flat_chain = chain[::50, :, :].reshape(-1, nd)


# In[18]:


from corner import corner


# In[27]:


nrpar = model.azr.config.n1


# In[28]:


fig = corner(flat_chain[:, :nrpar], show_titles=True)


# In[8]:


# files = ['AZUREOut_aa=1_R=1.out',
#          'AZUREOut_aa=1_R=2.out',
#          'AZUREOut_aa=1_TOTAL_CAPTURE.out']
# data = np.vstack([np.loadtxt('output/' + f) for f in files])
# np.savetxt('bare_uncertainties.txt', data[:, 6])


# In[ ]:




