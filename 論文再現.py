import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit

#パラメータ
L=100 #サイト数
A=1 #scaling factor
d=1 #discretisation length
w=1 #positive and controls the steepness of the curve around the horizon
alpha=A*d/w #dimensionless slope
l_h=50 #horizon at a distance l_h
l_i= 45#粒子の初期位置
t_f=9 #終状態の時刻
d_t=300 #時間分割数

#hopping amplitude
def kappa(l, l_h, alpha):
    return alpha*(l-l_h+0.5)/4

#ハミルトニアンと状態ベクトル、時間の初期化
H = np.zeros((L, L), dtype=complex)
psi = np.zeros((L,1), dtype=complex)
times = np.linspace(0, t_f, d_t)

#粒子は最初l=l_iにいる
psi[l_i-1,0] = 1  

# 共役転置関数
def hermitian(arr):
    return np.conjugate(arr.T)

#エンタングルメントエントロピー計算のために各時刻での波動関数,密度行列を格納するリスト
psi_list=[]
rho_out_list=[]
entropy_list=[]

########################################################################################################################################################

#FIG.3.(a)の再現
psi_abs=[]#各時刻での各サイトの存在確率を格納するリスト
#各サイトでの粒子存在確率を、各時刻で計算
for t in times:

    #ハミルトニアンの初期化
    H.fill(0)

    #時刻tでのハミルトニアンの生成
    for n in range(1,L):
        H[n-1, n] = -kappa(n, l_h, alpha)
        H[n, n-1] = -kappa(n, l_h, alpha)

    #時間発展ユニタリ演算子を生成し、時間発展させる
    U_t = scipy.linalg.expm(-1j * H * (t_f/d_t))
    psi = np.dot(U_t, psi)
    
    #エンタングルメントエントロピー計算のために各時刻での波動関数をリストに追加
    psi_list.append(psi)

    #状態ベクトルの各成分の大きさの２乗を成分として持つベクトルをリストに追加
    psi_abs.append(np.abs(psi)**2)

#plot
log_psi_abs=[]
log_psi_abs = np.log10(np.array(psi_abs)+1e-30)#確率をログスケールへ 
plt.figure(figsize=(6, 6))
heatmap=plt.imshow(log_psi_abs, extent=[0, L, times[0], times[-1]], aspect='auto', cmap='hot_r', origin='lower', vmin=-24, vmax=0)
plt.colorbar(heatmap, label='Log Probability Density',ticks=mticker.IndexLocator(base=3, offset=0))
plt.title('Logarithmic Scale Time Evolution of the State Vector')
plt.xlabel('Site Index')
plt.ylabel('Time')
plt.axvline(x=l_h, color='blue', linestyle='--', linewidth=2, label='Horizon')

plt.legend()
plt.show()

############################################################################################################################################################

#FIG.3.(b)の再現
t_b=6/alpha #この時刻で密度行列を計算する

#状態ベクトルの初期化
psi = np.zeros((L,1), dtype=complex)

#粒子の初期位置設定
psi[l_i-1,0] = 1  

#時間発展ユニタリ演算子を生成し、時間発展させる
U_t = scipy.linalg.expm(-1j * H * t_b)
psi = np.dot(U_t, psi)

# 状態ベクトルから密度行列を計算
rho = np.matmul(psi, hermitian(psi))

# Reduced density matricesを計算
rho_out = np.zeros((L-l_h, L-l_h), dtype=complex)#ブラックホール内部をトレースアウトした密度行列
for i in range(L-l_h):
    for j in range(L-l_h):
        rho_out[i,j]=psi[l_h+i,0]*np.conjugate(psi[l_h+j,0])

#時刻t_bでのハミルトニアン(horizonの外)の固有ベクトルと固有値を求める
H_out=H[l_h:,l_h:]#ブラックホール外部のハミルトニアン
eigenvalues, eigenvectors = np.linalg.eigh(H_out)

#負のエネルギーを切る
p_eigenvalues=[]#正の固有値を格納するリスト
p_eigenvectors=[]#正の固有値に対応する固有ベクトルを格納するリスト
for index,eigenvalue in enumerate(eigenvalues):
    if eigenvalue > 0:
        p_eigenvalues.append(eigenvalue)
        p_eigenvectors.append(eigenvectors[:,index])

p_eigenvectors=np.array(p_eigenvectors)


#E_nの確率を求める
p_probabilities=[] #各nに対するE_nの確率を格納するリスト
for p_eigenvector in p_eigenvectors:
    p_probabilities.append(np.vdot(hermitian(p_eigenvector),np.dot(rho_out,p_eigenvector)))

# 直線部分をフィットする関数(chat gpt)
def linear_fit(x, a, b):
    return a * x + b
    
#plot
#片対数へ変換
log_probabilities=[]
log_probabilities=np.log(np.array(p_probabilities)+1e-30)
# 直線部分を抜き出してフィット(chat gpt)
fit_indices = (np.array(p_eigenvalues) > 0.5) & (np.array(p_eigenvalues) < 5) # 適切な範囲を選択
fit_eigenvalues = np.array(p_eigenvalues)[fit_indices]
fit_log_probabilities = log_probabilities[fit_indices]
popt, pcov = curve_fit(linear_fit, fit_eigenvalues, fit_log_probabilities)
slope = popt[0]


plt.figure(figsize=(6,6))
plt.xticks([0,1,2,3,4,5])
plt.yticks([0,-6,-12,-18,-24])
plt.xlim(0,6)
plt.ylim(-24,0)
plt.grid()
plt.plot(p_eigenvalues, log_probabilities)
plt.plot(fit_eigenvalues, linear_fit(fit_eigenvalues, *popt), 'r--', label=f'Fit: slope={slope:.2f}')
plt.legend()
plt.title('Log Probability vs Eigenvalues')
plt.xlabel('Eigenvalues')
plt.ylabel('Log Probability')
plt.show()

print(f"The slope of the linear fit is: {slope}")
print(f"ホーキング温度は: {-1/slope}")

#######################################################################################################

#エンタングルメントエントロピーの時間変化をプロット
for psi_t in psi_list:
    # 各時刻でReduced density matricesを計算
    rho_out = np.zeros((L-l_h, L-l_h), dtype=complex)
    for i in range(L-l_h):
        for j in range(L-l_h):
            rho_out[i,j]=psi_t[l_h+i,0]*np.conjugate(psi_t[l_h+j,0])

    #エンタングルメントエントロピーの計算
    entropy=-1*np.trace(np.dot(rho_out,np.log(rho_out)))
    entropy_list.append(entropy)

#plot
plt.figure(figsize=(6,6))
#plt.xlim(0,6)
#plt.ylim(0,0.06)
plt.plot(times,entropy_list)
plt.show()