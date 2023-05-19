import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import math
from scipy.signal import hilbert

#%% 反射系数模块
def reflectivity_modling(N, modle_name, trace):
    import numpy as np

    # Reflection coefficient sampling point
    # modle one : one_reflectivity
    if modle_name == 'one_ref':
        ref = np.zeros((N, 1))
        ref[50] = 0.8
    elif modle_name == 'two_ref':
        ref = np.zeros((N, 1))
        ref[50] = 0.8
        ref[80] = -0.6
    elif modle_name == '500_ref':
        # Some parameter design
        num_ref = 2    # Number of single model pulses
        num_refk = 500    # Number of reflection coefficient models
        ref = np.zeros((N, num_refk))
        for k in range(num_refk):
            reflect0 = np.random.randint(8, 13, size=(2,))/10 * np.random.choice([-1, 1])    # 随机反射系数
            t = np.random.randint(0, N, size=num_ref)
            ref[t, k] = reflect0
            ref[-1, k] = 0
    elif modle_name == '1wedge1':
        # Wedge model
        # trace = 30
        ref = np.zeros((N, trace))
        ref[20, :] = 0.3
        for j in range(trace):
            ref[20+round((j)*1), j] = 0.3
    elif modle_name == '1wedge2':
        # Wedge model
        # trace = 30
        ref = np.zeros((N, trace))
        ref[40, :] = 0.3
        for j in range(4, trace):
            ref[40-3+round((j)*1), j] = 0.3

    return ref

#%%  平滑处理模块
def smooth5(x, pow):
    """
    用途：输入数据x，对该数据用相邻的5个点进行平滑
    参数说明：
        x - 输入数据
        y - 输出数据
        pow=2时，五点二次平滑方法
        pow=3时，五点三次平滑方法
    """
    import numpy as np
    N = len(x)
    y = np.zeros(N)
    if pow == 2:
        y[0] = (31 * x[0] + 9 * x[1] - 3 * x[2] - 5 * x[3] + 3 * x[4]) / 35
        y[1] = (9 * x[0] + 13 * x[1] + 12 * x[2] + 6 * x[3] - 5 * x[4]) / 35
        for k in range(2, N-2):
            y[k] = (-3 * x[k-2] + 12 * x[k-1] + 17 * x[k] + 12 * x[k+1] - 3 * x[k+2]) / 35
        y[-2] = (-5 * x[-5] + 6 * x[-4] + 12 * x[-3] + 13 * x[-2] + 9 * x[-1]) / 35
        y[-1] = (3 * x[-5] - 5 * x[-4] - 3 * x[-3] + 9 * x[-2] + 31 * x[-1]) / 35
    elif pow == 3:
        y[0] = (69 * x[0] + 4 * x[1] - 6 * x[2] + 4 * x[3] - x[4]) / 70
        y[1] = (2 * x[0] + 27 * x[1] + 12 * x[2] - 8 * x[3] + 2 * x[4]) / 35
        for k in range(2, N-2):
            y[k] = (-3 * x[k-2] + 12 * x[k-1] + 17 * x[k] + 12 * x[k+1] - 3 * x[k+2]) / 35
        y[-2] = (2 * x[-5] - 8 * x[-4] + 12 * x[-3] + 27 * x[-2] + 2 * x[-1]) / 35
        y[-1] = (-x[-5] + 4 * x[-4] - 6 * x[-3] + 4 * x[-2] + 69 * x[-1]) / 70
    else:
        raise ValueError("Error:该函数的平滑方法只有2次和3次！")
    return y

#%% 子波提取模块
import math
import numpy as np
from scipy.signal import hilbert
def fit_amplitude2min_zero_wavelet2D(synthetic, samt_syn, length, iter):
    # synthetic - 2D数组
    # samt_syn - 采样时间间隔
    # iter - 迭代次数
    # len - 滤波器长度

    slode_0 =np.hanning(23)
    slode=slode_0.reshape((-1,1))
    
    window = np.ones((synthetic.shape[0], 1))
    window[0:10] = slode[1:11]
    window[-10:] = slode[12:22]

    for k in range(synthetic.shape[1]):
        synthetic[:,k] = synthetic[:,k] * window[:,0]

   
        #synthetic[:, k] = synthetic[:, k] * win.flatten()
    #N_fft=np.power(2,np.ceil(np.log2(len(synthetic))).astype(int))*2
    N_fft = 2 ** np.ceil(np.log2(len(synthetic))).astype(int) * 2
    FS = np.abs(np.sum(np.fft.fft(synthetic, N_fft,axis=0), axis=1)) / synthetic.shape[1]
    f = np.arange(0, 1 / samt_syn, 1 / (N_fft * samt_syn))
    print(f.shape)
    Fw_esti = FS
    for k in range(iter):
        Fw_esti = smooth5(Fw_esti, 2)
        idx = np.where(Fw_esti < 0)
        Fw_esti[idx] = FS[idx]
        Fw_esti[0] = FS[0]
        Fw_esti[-1] = FS[-1]

    Fw_amp_esti = np.abs(Fw_esti)
    xxx=hilbert(np.log(Fw_amp_esti))
    #xxx = np.imag(np.log(Fw_amp_esti))
    ttt = (-1) * np.imag(xxx)

    wmin_esti = np.real(np.fft.ifft(Fw_amp_esti * np.exp(1j * ttt)))
    wmin_esti_cut = wmin_esti[0:length].T
    wzero_esti = np.real(np.fft.ifftshift(np.fft.fft(Fw_amp_esti)))
    M = math.floor(length / 2)
    #M=length//2
    id = np.argmax(np.abs(wzero_esti))
    wzero_esti_cut = wzero_esti[id - M:id + M + 1].T

    return wmin_esti_cut, wzero_esti_cut
#%% 加噪模块
def pnoise(data, nsr, seed=10):
    # Add random noise to the input data
    # Inputs:
    #   data: original data
    #   nsr: noise-to-signal ratio
    #   seed: seed for random number generation
    # Outputs:
    #   datans: data with added noise
    
    if nsr == 0:
        return data
    
    np.random.seed(seed)
    s_ener = np.linalg.norm(data)**2 # Signal energy
    zao = np.random.randn(*data.shape) # Generate Gaussian noise
    zao_ener = np.linalg.norm(zao)**2 # Noise energy
    factor = np.sqrt((s_ener / zao_ener) * nsr)
    noise = factor * zao
    datans = data + noise
    return datans

#%% 求解器A模块
def SolverFunc_A(seis, W, mu1, maxiter, tol):
    p = 1
    A = np.dot(W.T, W)
    inib = np.dot(W.T, seis)
    r = inib.copy()
    for k in range(maxiter):
        r1 = r.copy()
        Q = mu1 * np.diag(1/((np.abs(r))**(2-p) + np.finfo(float).eps)) # p
        Matrix = A + Q
        G = np.linalg.solve(Matrix, W.T)
        r = np.linalg.solve(Matrix, inib)
        r2 = r.copy()
        if np.sum(np.abs(r2 - r1)) / np.sum(np.abs(r1) + np.abs(r2)) < 0.5 * tol:
            break
    return r

#%% 求解器B模块
def SolverFunc_B(seis, W, mu1, maxiter, tol):
    p = 0.8
    # tol = 10e-10
    A = np.dot(W.T, W)
    inib = np.dot(W.T, seis)
    r = inib.copy()
    for k in range(maxiter):
        r1 = r.copy()
        Q = mu1 * np.diag(1/((np.abs(r))**(2-p) + np.finfo(float).eps)) # p
        Matrix = A + Q
        G = np.linalg.solve(Matrix, W.T)
        r = np.linalg.solve(Matrix, inib)
        r2 = r.copy()
        if np.sum(np.abs(r2 - r1)) / np.sum(np.abs(r1) + np.abs(r2)) < 0.5 * tol:
            break
    return r

#%% main
# set font properties
FONTSIZE = 18
FONTNAME = 'Times New Roman'
LINEWIDTH = 2

# Set parameters
N = 60        # Number of reflection coefficient sample points
trace = 30     # Number of model traces

#%% Generate reflectivity model
modle_name = '1wedge1'
ref = reflectivity_modling(N, modle_name, trace)
#%% 合成子波
dt=0.001 #1ms
fm=30
trun_time=0.04
t = np.arange(-trun_time, trun_time + dt, dt)
t = t.reshape((1,-1)) #转为行
w = (1 - 2 * (np.pi * fm * t) ** 2) * np.exp(-(np.pi * fm * t) ** 2)
plt.plot(w.T)
plt.show()
X=np.zeros((len(ref)-1,1))
d=np.append(w.T,X,axis=0)
nWaveSampPoint = len(w[0])
W_temp = toeplitz(d, np.zeros(len(ref)))

WW = W_temp[(nWaveSampPoint - 1) // 2 : - (nWaveSampPoint - 1) // 2, :]
#%% 合成地震记录
# Convolution, reverse, shift, multiply, and sum
seis = np.dot(WW, ref)
plt.imshow(seis)
plt.show()
#%% 加噪
#noise = np.random.normal(0,0.1,seis.shape)
#seis=seis + noise

# 0 is the mean of the normal distribution you are choosing from
# 1 is the standard deviation of the normal distribution
# 100 is the number of elements you get in array noise
seis=pnoise(seis,0.1)

#%%  参数设计
mu1=0.05
maxiter = 5
tol = 10e-20
#%% 情况选择
flag_wavelet = 2
if flag_wavelet == 1:
    dt=0.001 #1ms
    fm=30
    trun_time=0.04
    t = np.arange(-trun_time, trun_time + dt, dt)
    t = t.reshape((1,-1)) #转为行
    w = (1 - 2 * (np.pi * fm * t) ** 2) * np.exp(-(np.pi * fm * t) ** 2)
    X=np.zeros((N-1,1))
    d=np.append(w.T,X,axis=0)
    nWaveSampPoint = len(w[0])
    W_temp = toeplitz(d, np.zeros(len(ref)))
    WW = W_temp[(nWaveSampPoint - 1) // 2 : - (nWaveSampPoint - 1) // 2, :]
else:
    # 子波提取
    # Estimate wavelet  移植部分
    L_w = 101
    wmin_esti, wzero_esti = fit_amplitude2min_zero_wavelet2D(seis, dt, L_w, 100)
    wavelet = wzero_esti / max(wzero_esti)
    wavelet=wavelet.reshape((len(wavelet), 1))
    plt.plot(wavelet)
    plt.show()
    X=np.zeros((len(ref)-1,1))
    d=np.append(wavelet,X,axis=0)
    nWaveSampPoint = len(wavelet)
    W_temp = toeplitz(d, np.zeros(len(ref)))
    WW = W_temp[(nWaveSampPoint - 1) // 2 : - (nWaveSampPoint - 1) // 2, :]
#%% 求解器1
r_inv1 = np.zeros(ref.shape)
for i in range(seis.shape[1]):
    r_inv1[:, i] = SolverFunc_A(seis[:, i], WW, mu1, maxiter, tol)
plt.imshow(r_inv1)
plt.show()
#%% 求解器2
r_inv2 = np.zeros(ref.shape)
for i in range(seis.shape[1]):
    r_inv2[:, i] = SolverFunc_B(seis[:, i], WW, mu1, maxiter, tol)
plt.imshow(r_inv2)
plt.show()