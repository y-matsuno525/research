import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.sparse.linalg import expm
import math

#パラメータ
L=300 #サイト数は2L+1
d=0.1 #discretisation length
alpha=10 #dimensionless slope
n_0=int(-1*2/d)#粒子の初期位置
t_f=4 #終状態の時刻
d_t=200 #時間分割数

#hopping amplitude
def kappa(n,alpha):
    return alpha*np.tanh((n-0.5)*d)/(4*d)

#ハミルトニアンと状態ベクトル、時間の初期化
H = np.zeros((2*L+1, 2*L+1), dtype=complex)
psi = np.zeros((2*L+1,1), dtype=complex)
times = np.linspace(0, t_f, d_t)

#粒子は最初n_0にいる
psi[L+n_0,0] = 1  

# 共役転置関数
def hermitian(arr):
    return np.conjugate(arr.T)
###############################################################################################################################################

##test関数

def is_hermitian(matrix):
    
    return np.allclose(matrix, np.conj(matrix.T),atol=1e-100)

def is_unitary(matrix):
    
    # 単位行列を生成
    identity_matrix = np.eye(matrix.shape[0])
    
    # Uの共役転置行列とUの積が単位行列かどうかを確認(atolは許容する差分)
    return np.allclose(np.dot(hermitian(matrix), matrix), identity_matrix,atol=1e-13)

################################################################################################################################################
##Rydberg atomsの相互作用項の係数を計算

f_inf = float('inf') #無限に離れた距離を表現するため
c=1 #c_3の比例定数

#最近接原子間距離を格納
d_matrix=np.zeros((2*L+1,2*L+1))
for n in range(1,2*L+1):
    kappa_n = kappa(n-300,alpha)
    if n < 301:
        d_matrix[n,n-1]=math.pow(-4*c/(kappa_n),1/3)
        d_matrix[n-1,n]=math.pow(-4*c/(kappa_n),1/3)
    else:
        d_matrix[n,n-1]=math.pow(-4*c/(-1*kappa_n),1/3)
        d_matrix[n-1,n]=math.pow(-4*c/(-1*kappa_n),1/3)
        
#NNNの距離を格納
for n in range(2*L-1):
    d_matrix[n,n+2]=d_matrix[n,n+1]+d_matrix[n+1,n+2]
    d_matrix[n+2,n]=d_matrix[n,n+1]+d_matrix[n+1,n+2]

#NNNによるHの行列要素を格納(couplingのc)
c_matrix=np.zeros((2*L+1,2*L+1))
for n in range(2*L-2):
    c_matrix[n,n+2]=-4*c/(d_matrix[n,n+2])**3
    c_matrix[n+2,n]=-4*c/(d_matrix[n,n+2])**3
    

#################################################################################################################################################

#FIG.1.(a)の再現(Hawking温度)

#エンタングルメントエントロピー計算のために各時刻での波動関数,密度行列を格納するリスト
entropy_list=[]

#ハミルトニアンの生成
for n in range(1,2*L+1):
    if n != 2*L:
        if n < 301:
            H[n-1, n] = kappa(n-300,alpha)
            H[n, n-1] = kappa(n-300,alpha)
        else:
            H[n-1, n] = -kappa(n-300,alpha)
            H[n, n-1] = -kappa(n-300,alpha)
        if n != 2*L-1:
            H[n-1,n+1]= c_matrix[n-1,n+1]
            H[n+1,n-1]= c_matrix[n+1,n-1]
    else:
        H[n-1, n] = 0
        H[n, n-1] = 0
        
#時間発展ユニタリ演算子を生成し、時間発展させる
U_t = expm(-1j * H * 4)
print("ハミルトニアンがエルミートか："+str(is_hermitian(H)))
print("時間発展演算子がユニタリか："+str(is_unitary(U_t)))
psi = np.dot(U_t, psi)
psi /= np.linalg.norm(psi)

# Reduced density matricesを計算
rho_out = np.zeros((L+1, L+1), dtype=complex)#ブラックホール内部をトレースアウトした密度行列

for i in range(1,L+2):
    for j in range(1,L+2):
        if (i == L+1) or (j == L+1):#(L+1,L+1)成分はあとで代入
            rho_out[i-1,j-1]=0
        else:
            rho_out[i-1,j-1]=psi[L+i,0]*np.conjugate(psi[L+j,0])

for i in range(L): #rho_outの(L+1,L+1)成分を代入
    rho_out[L, L] += (psi[i] * np.conjugate(psi[i])).real


        
#時刻t_bでのハミルトニアン(horizonの外)の固有ベクトルと固有値を求める
H_out=np.zeros((L+1, L+1), dtype=complex)#ブラックホール外部のハミルトニアン
for i in range(1,L+2):
    for j in range(1,L+2):
        if (i == L+1) or (j == L+1):
            H_out[i-1,j-1]=0
        else:
            H_out[i-1,j-1]=H[L+1:,L+1:][i-1,j-1]

eigenvalues, eigenvectors = np.linalg.eigh(H_out)

#負のエネルギーを切る
p_eigenvalues=[]#正の固有値を格納するリスト
p_eigenvectors=[]#正の固有値に対応する固有ベクトルを格納するリスト
for index,eigenvalue in enumerate(eigenvalues):
    if (eigenvalue > 0)&(eigenvalue < 100):
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
fit_indices = (np.array(p_eigenvalues) > 0) & (np.array(p_eigenvalues) < 15) # 適切な範囲を選択
fit_eigenvalues = np.array(p_eigenvalues)[fit_indices]
fit_log_probabilities = log_probabilities[fit_indices]
popt, pcov = curve_fit(linear_fit, fit_eigenvalues, fit_log_probabilities)
slope = popt[0]

plt.figure(figsize=(8,7))
plt.xticks([0,5,10,15])
plt.yticks([-20,-15,-10,-5,0])
#plt.xlim(0,15)
#plt.ylim(-30,0)
plt.plot(p_eigenvalues, log_probabilities)
plt.plot(fit_eigenvalues, linear_fit(fit_eigenvalues, *popt), 'r--', label=f'Fit: slope={slope:.2f}')
plt.legend()
plt.title('Log Probability vs Eigenvalues')
plt.xlabel('Eigenvalues')
plt.ylabel('Log Probability')
plt.show()

print(f"The slope of the linear fit is: {slope}")
print(f"ホーキング温度は: {-1/slope}")

######################################################################################################

#FIG.1.(b)の再現

#粒子の初期化
t_max_eigenvalue=0

for t in times:

    psi = np.zeros((2*L+1,1), dtype=complex)
    
    psi[L+n_0,0] = 1
    #時間発展
    U_t = expm(-1j * H * t)
    psi = np.dot(U_t, psi)
    psi /= np.linalg.norm(psi)

    # Reduced density matricesを計算
    for i in range(1,L+2):
        for j in range(1,L+2):
            if (i == L+1) or (j == L+1):#(L+1,L+1)成分はあとで代入
                rho_out[i-1,j-1]=0
            else:
                rho_out[i-1,j-1]=psi[L+i,0]*np.conjugate(psi[L+j,0])

    for i in range(L): #rho_outの(L+1,L+1)成分を代入
        rho_out[L, L] += (psi[i] * np.conjugate(psi[i])).real


        

    #エンタングルメントエントロピーの計算&格納
    eigenvalues,_=np.linalg.eig(rho_out)
    entropy=0

    #正、０、負の固有値の数を記録する変数
    #count_p=0
    #count_0=0
    #count_n=0
    
    #max_eigenvalue=max(eigenvalues)
    #if max_eigenvalue > t_max_eigenvalue:
        #t_max_eigenvalue = max_eigenvalue

    for eigenvalue in eigenvalues:
        if eigenvalue > 0:
            entropy += -1*(eigenvalue*np.log(eigenvalue))
        #if eigenvalue < 0:
            #count_n += 1
        #elif eigenvalue == 0:
            #count_0 += 1
        #else:
            #count_p += 1
            
            
            
    #print("正：０：負は"+str(count_p)+":"+str(count_0)+":"+str(count_n)+"です。")
        
    entropy_list.append(entropy)

    print(str(t/t_f*100)+"%")
    

#plot
plt.figure(figsize=(6,6))
plt.grid()
plt.plot(times,entropy_list)
plt.show()
print(t_max_eigenvalue)