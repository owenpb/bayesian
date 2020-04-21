# Bayesian Inference Assignment

import numpy as np
import matplotlib.pyplot as plt

# If P(x|a) = N(a) a^{-x^2/2} then N(a) = sqrt{ln(a)/(2*pi)} for normalization

# Bayes theorem: P(a|x) = P(x|a) P(a)/P(x)

# With n samples, we get:
# P(a|{x}) = P({x}|a) P(a)/P({x})
# where:
# P({x}|a) = N(a)^n a^(-1/2 \sum_i x_i^2)

# P({x}|a) is called the likelihood
# P(a|{x}) is called the posterior
# P(a) is called the prior. Take to be uniform: P(a) = 1?
# Then: P(a|{x}) = N(a)^n a^(-1/2 \sum_i x_i^2) / P({x})


def normfactor(a):
    return np.sqrt(np.log(a) / (2 * np.pi))


def gauss(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x ** 2))


np.random.seed(2)

samples = []
# Generate 100 samples:
for i in range(100):
    # (np.random.normal())
    samples.append(np.random.normal())

exponent = -0.5 * np.sum(np.square(samples))

# print(exponent)

a_values = np.linspace(0, 5, 500)
like_values = []

for a in a_values:
    like_values.append((normfactor(a) ** 100) * (a ** exponent))

# print(a_values)
# print(like_values)

# print(gauss(0))

px = 1
for k in samples:
    px = px * gauss(k)
    # print(px)
# print(px)

post_values = like_values / px

print(post_values)

plt.plot(a_values, post_values)
plt.xlabel("a")
plt.ylabel("P(a|{x})")
plt.show()

# Create Markov Chain

# Step 1: Pick random starting value of a (in the range -10 to 10)
a_init = (np.random.rand() * 20) + (-10)
# print(a_init)

# Step 2: Set number of timesteps, deviation sigma in the generator,
# and enter the loop:
timesteps = 50000
sigma = 0.1

a_history = []
time_history = []

a_curr = a_init
for t in range(timesteps):
    a_cand = np.random.normal(a_curr, sigma)
    p_cand = (normfactor(a_cand) ** 100) * (a_cand ** exponent)
    p_curr = (normfactor(a_curr) ** 100) * (a_curr ** exponent)
    ratio = p_cand / p_curr
    acceptance = min(1, ratio)
    u = np.random.rand()
    if u <= acceptance:
        a_curr = a_cand

    a_history.append(a_curr)
    time_history.append(t)

plt.plot(time_history, a_history)
plt.show()

num_bins = 100
n, bins, patches = plt.hist(a_history, num_bins, facecolor="blue", alpha=0.5)
plt.show()
