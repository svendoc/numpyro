# Experiment: Comp MAP, SVGD, SMI and MFVI

from numpyro import sample, prng_key, param, deterministic
from numpyro.distributions import Normal, Delta, HalfNormal
from numpyro.distributions.constraints import softplus_positive, positive
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

T = 10

with seed(rng_seed=100):
    samples = Predictive(model, num_samples = 1, return_sites={'x0', 'x', 'y'})(prng_key(), T, None)

x = jnp.concatenate([samples['x0'][:, None], samples['x']], axis=1)[0]
y = jnp.concatenate([jnp.array([jnp.nan]), samples['y'][0]])

plt.plot(x, label='True x (unobserved)', linewidth=3, alpha=.8, color = 'blue')
plt.plot(y, 'rx', label='observed y')

def guide(state, y):
    x = param('xp', lambda key: Normal().sample(key, (T-1,)))
    sample('x', Delta(x))

MAP = SVI(model, guide, Adagrad(1.), Trace_ELBO())
with seed(rng_seed=0):
    res = MAP.run(prng_key(), 1, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'], res.params['xp']])
plt.plot(x_app, label='MAP x (1 step)', color='black', linewidth=5)

with seed(rng_seed=0):
    res = MAP.run(prng_key(), 3_000, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'], res.params['xp']])
plt.plot(x_app, label='MAP x (3k steps)', color='black', linewidth=3)

def guide(state, y):
    x = param('xp', lambda key: Normal().sample(key, (T-1,)))
    sample('x', Normal(x, 1.))

smi = SVI(model, guide, Adagrad(1.), Trace_ELBO())
with seed(rng_seed=0):
    res = smi.run(prng_key(), 1, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'], res.params['xp']])
plt.plot(x_app, label='mfvi(loc) x (1 step)', color='yellow', linewidth=3)

with seed(rng_seed=0):
    res = smi.run(prng_key(), 3000, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'], res.params['xp']])
plt.plot(x_app, '--', label='mfvi(loc) x (3k steps)', color='yellow', linewidth=3)


def guide(state, y):
    x = param('xp', lambda key: Normal().sample(key, (T-1,)))
    scale = param('scale', lambda key: HalfNormal().sample(key, (T-1,)), constraint=positive)
    sample('x', Normal(x, scale))

from numpyro.infer.autoguide import AutoNormal

# mfvi = SVI(model, AutoNormal(model), Adagrad(1.), Trace_ELBO())
smi = SVI(model, guide, Adagrad(1.), Trace_ELBO())
with seed(rng_seed=0):
    res = smi.run(prng_key(), 1, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'], res.params['xp']])
plt.plot(x_app, label='mfvi(loc, scale) x (1 step)', color='red', linewidth=1)

with seed(rng_seed=0):
    res = smi.run(prng_key(), 3000, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'], res.params['xp']])
plt.plot(x_app, '--', label='mfvi(loc, scale) x (3k steps)', color='red', linewidth=3)

def guide(state, y):
    x = param('xp', lambda key: Normal().sample(key, (T-1,)))
    sample('x', Normal(x, 1.))

svgd = SteinVI(model, guide, Adagrad(1.), RBFKernel(), num_stein_particles=3)
with seed(rng_seed=0):
    res = svgd.run(prng_key(), 1, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'].repeat(3).reshape(3,1), res.params['xp']], axis=1)
plt.plot(x_app[0], label='svgd1 x (1 step)', color='purple', linewidth=3)
plt.plot(x_app[1], label='svgd2 x (1 step)', color='purple', linewidth=3)
plt.plot(x_app[2], label='svgd3 x (1 step)', color='purple', linewidth=3)

with seed(rng_seed=0):
    res = svgd.run(prng_key(), 3000, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'].repeat(3).reshape(3,1), res.params['xp']], axis=1)
plt.plot(x_app[0], label='svgd1 x (3k step)',  color='purple', linewidth=3)
plt.plot(x_app[1], label='svgd2 x (3k step)',  color='purple', linewidth=3)
plt.plot(x_app[2], label='svgd3 x (3k step)',  color='purple', linewidth=3)

def guide(state, y):
    x = param('xp', lambda key: Normal().sample(key, (T-1,)))
    scale = param('scale', lambda key: HalfNormal().sample(key, (T-1,)), constraint=positive)
    sample('x', Normal(x, scale))

smi = SteinVI(model, guide, Adagrad(1.), RBFKernel(), num_stein_particles=3)
with seed(rng_seed=0):
    res = smi.run(prng_key(), 1, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'].repeat(3).reshape(3,1), res.params['xp']], axis=1)
plt.plot(x_app[0], label='smi1 x (1 step)', color='orange', linewidth=1)
plt.plot(x_app[1], label='smi2 x (1 step)', color='orange', linewidth=1)
plt.plot(x_app[2], label='smi3 x (1 step)', color='orange', linewidth=1)

with seed(rng_seed=0):
    res = smi.run(prng_key(), 3000, T, samples['y'][0])

x_app = jnp.concatenate([samples['x0'].repeat(3).reshape(3,1), res.params['xp']], axis=1)
plt.plot(x_app[0], label='smi1 x (3k step)',linestyle='--', color='orange', linewidth=3)
plt.plot(x_app[1], label='smi2 x (3k step)',linestyle='--', color='orange', linewidth=3)
plt.plot(x_app[2], label='smi3 x (3k step)',linestyle='--', color='orange', linewidth=3)

plt.xlabel('time')
plt.ylabel('location')
plt.legend()
plt.show(block=True)