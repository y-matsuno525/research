import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

L = 50  
d = 1
alpha = 1
n_h = L
t_f=9
t_d=200
v = 0

def kappa(n, d, alpha, time):
    return alpha * (n - (n_h - v * time) + 0.5) / 4

H = np.zeros((2*L+1, 2*L+1), dtype=complex)

psi = np.zeros(2*L+1, dtype=complex)
psi[45] = 1  

times = np.linspace(0, t_f, t_d)  

psi_f = []
prob_density_n_h=[]

for t in times:
    
    for n in range(1, 2*L+1):
        H[n-1, n] = -kappa(n, d, alpha, t)
        H[n, n-1] = -kappa(n, d, alpha, t)
    
    U_t = scipy.linalg.expm(-1j * H * (t_f/t_d))
    psi = np.dot(U_t, psi)
    
    psi_f.append(np.abs(psi)**2)

    prob_density_n_h.append(np.abs(psi[n_h])**2)

psi_f = np.array(psi_f)



# 確率密度の時間変化をプロット
plt.figure(figsize=(10, 6))
plt.plot(times, prob_density_n_h, label='Site 50')
plt.title('Time Evolution of Probability Density at Site 50')
plt.xlabel('Time')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# ログスケールでの時間発展のプロット
log_psi_f = np.log10(psi_f + 1e-21)  

plt.figure(figsize=(6, 6))
plt.imshow(log_psi_f, extent=[0, 2*L+1, times[0], times[-1]], aspect='auto', cmap='hot_r', origin='lower', vmin=-20, vmax=0)
plt.colorbar(label='Log Probability Density')
plt.title('Logarithmic Scale Time Evolution of the State Vector')
plt.xlabel('Site Index')
plt.ylabel('Time')
plt.show()
