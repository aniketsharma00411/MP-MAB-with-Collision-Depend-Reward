import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


num_arms = 10
sigma = 0.2


def no_collision_rewards(arm):
    mean_rewards = np.linspace(0.3, 0.84, num_arms)
    distribution = tfp.distributions.GeneralizedNormal(
        mean_rewards[arm],
        (2**0.5)*sigma,
        8,
        name='GeneralizedNormal'
    )

    arm_rewards = distribution.sample(sample_shape=(1), seed=42)
    arm_rewards = float(arm_rewards[0])
    if arm_rewards < 0:
        yield 0
    elif arm_rewards > 1:
        yield 1
    else:
        yield arm_rewards


def collision_rewards(arm):
    mean_rewards = [0.1]*num_arms
    distribution = tfp.distributions.GeneralizedNormal(
        mean_rewards[arm],
        (2**0.5)*sigma,
        8,
        name='GeneralizedNormal'
    )

    arm_rewards = distribution.sample(sample_shape=(1), seed=42)
    arm_rewards = float(arm_rewards[0])
    if arm_rewards < 0:
        yield 0
    elif arm_rewards > 1:
        yield 1
    else:
        yield arm_rewards
