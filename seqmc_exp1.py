# Experiment: Comp MAP post for p(y|x) and p(y|x+1)

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

def evolvep1(state,y):
    x, t = state
    x = sample('x', Normal(x, 1.))
    t = t + 1
    y = sample('y', Normal(x+1, .2), obs=y)
    return (x, t), y

def modelp1(T, y):
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
        evolvep1, # sample p(x_{t+1}|x_t)p(y_{t+1}|x_{t+1})
        (x0, t0), 
        y,
        length = T - 1  # we already have x0
    )

def evolve(state,y):
    x, t = state
    x = sample('x', Normal(x, 1.))
    t = t + 1
    y = sample('y', Normal(x, .2), obs=y)
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

T = 10

# 
with seed(rng_seed=100):
    samples = Predictive(modelp1, num_samples = 1, return_sites={'x0', 'x', 'y'})(prng_key(), T, None)

x = jnp.concatenate([samples['x0'][:, None], samples['x']], axis=1)[0]
y = jnp.concatenate([jnp.array([jnp.nan]), samples['y'][0]])

plt.plot(x, label='True x (unobserved)', linewidth=3, alpha=.8, color = 'blue')
plt.plot(y, 'rx', label='observed y(p1)')
plt.plot(y-1, 'go', label='observed y')

def guide(state, y):
    x = param('xp', lambda key: Normal().sample(key, (T-1,)))
    sample('x', Delta(x))

MAP = SVI(modelp1, guide, Adagrad(1.), Trace_ELBO())
with seed(rng_seed=0):
    res = MAP.run(prng_key(), 1, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'], res.params['xp']])
plt.plot(x_app, label='MAP(p1) x (1 step)', color='black', linewidth=5)

with seed(rng_seed=0):
    res = MAP.run(prng_key(), 3000, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'],res.params['xp']])
plt.plot(x_app, label='MAP(p1) x (3k steps)', linewidth=3)



MAP = SVI(model, guide, Adagrad(1.), Trace_ELBO())
with seed(rng_seed=0):
    res = MAP.run(prng_key(), 1, T, samples['y'][0] - 1)

x_app = jnp.concatenate([samples['x0'], res.params['xp']])
plt.plot(x_app, label='MAP x (1 step)', color='yellow', linewidth=2)

with seed(rng_seed=0):
    res = MAP.run(prng_key(), 3000, T, samples['y'][0] - 1)

x_app = jnp.concatenate([samples['x0'],res.params['xp']])
plt.plot(x_app, '--', label='MAP x (3k steps)', linewidth=3)
plt.hlines(y[1:].mean(),       xmin=0, xmax=T, colors = 'red', label='y(p1) mean')
plt.hlines((y[1:] - 1).mean(), xmin=0, xmax=T, colors='green', label='y mean')
plt.hlines(x[1:].mean(),       xmin=0, xmax=T, colors='blue', label='x mean')

print('MAP(p1) x', x_app[-1], 'y mean', y[1:].mean(), 'x mean', x[1:].mean())
print('MAP x', x_app[-1], 'y mean', (y[1:] -1).mean(), 'x mean', x[1:].mean())

plt.legend()
plt.show(block=True)