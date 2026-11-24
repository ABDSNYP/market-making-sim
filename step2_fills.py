import numpy as np
import matplotlib.pyplot as plt

# Order-flow model: market orders hit a quote at distance delta from the mid
# as a Poisson process with intensity  lambda(delta) = A * exp(-k * delta).
A, k = 140.0, 1.5
T, dt = 1.0, 0.005
n_steps = int(T / dt)
rng = np.random.default_rng(1)


def fill_prob(delta):
    """Chance our quote gets hit during one small time step."""
    return np.minimum(A * np.exp(-k * delta) * dt, 1.0)


# Try a range of quote distances and simulate 2,000 sessions for each
deltas = np.linspace(0.05, 3.0, 40)
n_sessions = 2000
sim_fills = []
for d in deltas:
    hits = rng.random((n_steps, n_sessions)) < fill_prob(d)   # True where we got filled
    sim_fills.append(hits.sum(axis=0).mean())                 # avg fills per session
sim_fills = np.array(sim_fills)

theory_fills = A * np.exp(-k * deltas) * T    # expected fills = intensity x time
edge = sim_fills * deltas                     # what we earn: fills x distance from mid

best = deltas[np.argmax(edge)]
print(f"best distance (simulated): {best:.3f}")
print(f"best distance (theory 1/k): {1 / k:.3f}")
print(f"expected edge per side at 1/k: {A * np.exp(-1) / k * T:.1f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(deltas, sim_fills, "o", ms=4, label="simulated")
a1.plot(deltas, theory_fills, "-", label="theory  A·e^(-kδ)·T")
a1.set_xlabel("Quote distance from mid (δ)")
a1.set_ylabel("Fills per session")
a1.set_title("Quote tighter, get filled more")
a1.legend(frameon=False)

a2.plot(deltas, edge, "o-", ms=4)
a2.axvline(1 / k, ls="--", color="gray")
a2.text(1 / k + 0.05, edge.max() * 0.5, "δ = 1/k", color="gray")
a2.set_xlabel("Quote distance from mid (δ)")
a2.set_ylabel("Edge earned per session (fills × δ)")
a2.set_title("...but each fill earns less: there's a sweet spot")

for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("step2_fills.png", dpi=150)
print("saved step2_fills.png")

