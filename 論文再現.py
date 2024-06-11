import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

#パラメータ
L=100 #サイト数
A=1 #scaling factor
d=1 #discretisation length
w=1 #positive and controls the steepness of the curve around the horizon
alpha=A*d/w #dimensionless slope
l_h=50 #horizon at a distance l_h
t_f=9 #終状態の時刻
d_t=200 #時間分割数

#hopping amplitude
def kappa(l, l_h, alpha):
    return alpha * (l-l_h+0.5)/4

#ハミルトニアンと状態ベクトル、時間の初期化
H = np.zeros((L, L), dtype=complex)
psi = np.zeros(L, dtype=complex)
times = np.linspace(0, t_f, d_t)

#粒子は最初l=45にいる
psi[45] = 1  


#FIG.3.(a)の再現
psi_abs=[]#各時刻での各サイトの存在確率を格納するリスト
#各サイトでの粒子存在確率を、各時刻で計算
for t in times:

    #ハミルトニアンの初期化
    H.fill(0)

    #時刻tでのハミルトニアンの生成
    for n in range(1, L):
        H[n-1, n] = -kappa(n, l_h, alpha)
        H[n, n-1] = -kappa(n, l_h, alpha)

    #時間発展ユニタリ演算子を生成し、時間発展させる
    U_t = scipy.linalg.expm(-1j * H * (t_f/d_t))
    psi = np.dot(U_t, psi)

    #状態ベクトルの各成分の大きさの２乗を成分として持つベクトルをリストに追加
    psi_abs.append(np.abs(psi)**2)

#plot
log_psi_abs=[]
log_psi_abs = np.log10(np.array(psi_abs)+1e-30)#確率をログスケールへ 
plt.figure(figsize=(6, 6))
plt.imshow(log_psi_abs, extent=[0, L, times[0], times[-1]], aspect='auto', cmap='hot_r', origin='lower', vmin=-23, vmax=0)
plt.colorbar(label='Log Probability Density')
plt.title('Logarithmic Scale Time Evolution of the State Vector')
plt.xlabel('Site Index')
plt.ylabel('Time')
plt.show()