import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# パラメータの設定
L = 100  # 以前よりも小さいLで試みる
d = 0.1
alpha = 10
n_h = L

# ホッピング振幅の計算
def kappa(n, d, alpha):
    return alpha * np.tanh((n - n_h - 0.5) * d) / (4 * d)

# ハミルトニアン行列の定義
H = np.zeros((2*L+1, 2*L+1), dtype=complex)
for n in range(1, 2*L+1):
    H[n-1, n] = -kappa(n, d, alpha)
    H[n, n-1] = -kappa(n, d, alpha)

# 初期状態の定義
psi_0 = np.zeros(2*L+1, dtype=complex)
psi_0[0] = 1  # 初期状態を真ん中に変更

# 時刻のリストを設定
times = np.linspace(0, 20, 100)  # より多くの時間ステップを設定

# 各時刻での状態ベクトルを計算
psi_ts = []
for t in times:
    U_t = scipy.linalg.expm(-1j * H * t)
    psi_t = np.dot(U_t, psi_0)
    psi_ts.append(np.abs(psi_t))

# アニメーションの設定
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(np.abs(psi_ts[0]))
ax.set_xlim(0, 2*L)
ax.set_ylim(0, 1)
ax.set_xlabel('Site Index')
ax.set_ylabel('|ψ(t)|')
ax.set_title('Time Evolution of State Vector |ψ(t)|')

def update(num, psi_ts, line):
    line.set_ydata(psi_ts[num])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(times), fargs=[psi_ts, line], interval=50, blit=True)
plt.show()
