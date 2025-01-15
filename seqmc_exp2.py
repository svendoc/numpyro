# Compute the Trace_ELBO and compare with the manual computation

from numpyro import sample, prng_key, param, deterministic
from numpyro.distributions import Normal, Delta
from numpyro.contrib.control_flow import scan
from jax import numpy as jnp
from numpyro.handlers import seed
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDelta
from numpyro.contrib.einstein import RBFKernel, SteinVI
from numpyro.optim import Adagrad
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS

from matplotlib import pyplot as plt
import matplotlib


def evolve(state,y):
    x, t = state
    x = sample('x', Normal(x, 1.))
    t = t + 1
    y = sample('y', Normal(x, .1), obs=y)
    return (x, t), y

def model(T, y):
    """ x is our unobserved latent state and y is our observations.

    :param T: total number of steps
    :param y: Observations, either None or (n,T) matrix
    """
    if y is not None:
        t, = y.shape
        assert t == (T-1)
    
    # Initial x (ie. x0)
    x0 = deterministic('x0', 0.)
    t0 = 0

    scan(
        evolve, # sample p(x_{t+1}|x_t)p(y_{t+1}|x_{t+1})
        (x0, t0), 
        y,
        length = T - 1  # we already have x0
    )

def guide(state, y):
    x = param('xp', lambda key: Normal().sample(key, (T-1,)))
    sample('x', Delta(x))

T = 10

with seed(rng_seed=100):
    samples = Predictive(model, num_samples = 1, return_sites={'x0', 'x', 'y'})(prng_key(), T, None)

x = jnp.concatenate([samples['x0'][:, None], samples['x']], axis=1)[0]
y = jnp.concatenate([jnp.array([jnp.nan]), samples['y'][0]])

with seed(rng_seed=100):
    params = {'xp': Normal().sample(prng_key(), (T-1,))}
    elbo = Trace_ELBO()
    print(elbo.loss(prng_key(), params, model, guide, T, y[1:]))

# We need to compute -(E_{x~q}[log p(x,y)] - E_{x~q}[log q(x)])
#
# q(z) = 1 if z==x else 0 
#
# p(x,y) = p(x_1)p(y_1|x_1) *
#          * prod_{i=2}^T p(x_i | x_{i-1})p(y_i | x_i) <=>
# log p(x,y) = log p(x_1) + p(y_1|x_1) + 
#              + sum_{i=2}^T log p(x_i | x_{i-1}) + log p(y_i | x_i)
#
# First notice that E_{z~q}[log q(z)] = 0. 
# We have that q(x) = 1 (or equivalently log q(x)=0) by definition.
# Since we can only sample x from q we can rewrite E_{z~q}[log q(z)] to log p(x) = 0 by the above.

# Because of the delta in the guide we know params['xp'] is our x in p(x,y).

# From evolve we have that log p(x1) and p(y1|x1) are

xp = jnp.concatenate([samples['x0'], params['xp']])
logp_x1 = Normal().log_prob(xp[1])
logp_y1 = Normal(xp[1], .1).log_prob(y[1])

# and that sum_{i=2}^T log p(x_i | x_{i-1}) + log p(y_i | x_i) is
logp_x2_T = [Normal(xp[i-1]).log_prob(xp[i]).item() for i in range(2, T)]
logp_y2_T = [Normal(xp[i], .1).log_prob(y[i]).item() for i in range(2, T)]

print(logp_x1 + logp_y1 + sum(logp_x2_T) + sum(logp_y2_T))

