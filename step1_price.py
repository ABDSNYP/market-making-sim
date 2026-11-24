import numpy as np
import matplotlib.pyplot as plt

S0, T, sigma, dt = 100.0, 1.0, 2.0, 0.005
n_steps = int(T / dt)          # 200 steps
rng = np.random.default_rng(0)

# TODO 1: random +1 / -1 moves
moves = rng.choice([-1, 1], size=n_steps)

# TODO 2: scale each move so volatility is consistent with the time step
increments = moves * sigma * np.sqrt(dt)

# TODO 3: price path, starting at S0 (201 points: time 0 plus 200 steps)
prices = np.insert(S0 + np.cumsum(increments), 0, S0)
print(len(prices), prices[:5])

# TODO 4: plot it
t = np.linspace(0, T, n_steps + 1)
plt.figure(figsize=(8, 4))
plt.plot(t, prices)
plt.xlabel("Time")
plt.ylabel("Mid price")
plt.title("Simulated mid-price (random walk)")
plt.tight_layout()
plt.savefig("step1_price.png", dpi=150)
print("saved step1_price.png")

# TODO 5: 1,000 paths at once, then check the spread of final prices
n_paths = 1000
all_moves = rng.choice([-1, 1], size=(n_steps, n_paths))
final_prices = S0 + (all_moves * sigma * np.sqrt(dt)).sum(axis=0)
print("std of final prices:", final_prices.std())
print("theory says:        ", sigma * np.sqrt(T))
