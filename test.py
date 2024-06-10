import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Parameters
L = 100  # Smaller value for visualization purposes
d = 0.1
alpha = 10
n_h = L

# Function to calculate hopping amplitude
def kappa(n, d, alpha):
    return alpha * np.tanh((n - n_h - 0.5) * d) / (4 * d)

# Define Hamiltonian matrix
H = np.zeros((2*L+1, 2*L+1), dtype=complex)
for n in range(1, 2*L+1):
    H[n-1, n] = -kappa(n, d, alpha)
    H[n, n-1] = -kappa(n, d, alpha)

# Define initial state vector
psi_0 = np.zeros(2*L+1, dtype=complex)
psi_0[0] = 1  # Central site for visualization

# List of times for evolution
times = np.linspace(0, 100, 300)  # More time points for a smooth animation

# Calculate the state vector at each time point
psi_ts = []
for t in times:
    U_t = scipy.linalg.expm(-1j * H * t)
    psi_t = np.dot(U_t, psi_0)
    psi_ts.append(np.abs(psi_t)**2)  # Store the probability density

# Convert to numpy array for easier slicing
psi_ts = np.array(psi_ts)

# Plotting the state evolution similar to FIG.3(a) from the paper
plt.figure(figsize=(10, 6))
plt.imshow(psi_ts.T, extent=[times[0], times[-1], 0, 2*L+1], aspect='auto', cmap='hot')
plt.colorbar(label='Probability Density')
plt.title('Time Evolution of the State Vector')
plt.xlabel('Time')
plt.ylabel('Site Index')
plt.show()
