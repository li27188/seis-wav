# 反射系数模块
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
        ref[80, :] = 0.3
        for j in range(trace):
            ref[80+round((j)*1), j] = 0.3
    elif modle_name == '1wedge2':
        # Wedge model
        # trace = 30
        ref = np.zeros((N, trace))
        ref[40, :] = 0.3
        for j in range(4, trace):
            ref[40-3+round((j)*1), j] = 0.3

    return ref


# 平滑处理模块
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



# 子波提取模块
import numpy as np
def fit_amplitude2min_zero_wavelet2D(synthetic, samt_syn, length, iter):
    # synthetic - 2D数组
    # samt_syn - 采样时间间隔
    # iter - 迭代次数
    # len - 滤波器长度

    slode_0 =np.hanning(21)
    slode=slode_0.reshape((-1,1))
    print(slode)
    window = np.ones((synthetic.shape[0], 1))
    window[0:10] = slode[0:10]
    window[-10:] = slode[11:21]
    for k in range(synthetic.shape[1]):
        synthetic[:,k] = synthetic[:,k] * window[:,0]
        #synthetic[:, k] = synthetic[:, k] * win.flatten()
    N_fft = 2 ** np.ceil(np.log2(len(synthetic))).astype(int) * 2

    FS = np.abs(np.sum(np.fft.fft(synthetic, N_fft), axis=1)) / synthetic.shape[1]
    f = np.arange(0, 1 / samt_syn, 1 / (N_fft * samt_syn))

    Fw_esti = FS
    for k in range(iter):
        Fw_esti = smooth5(Fw_esti, 2)
        idx = np.where(Fw_esti < 0)
        Fw_esti[idx] = FS[idx]
        Fw_esti[0] = FS[0]
        Fw_esti[-1] = FS[-1]

    Fw_amp_esti = np.abs(Fw_esti)
    xxx = np.imag(np.log(Fw_amp_esti))
    ttt = (-1) * np.imag(xxx)
    wmin_esti = np.real(np.fft.ifft(Fw_amp_esti * np.exp(1j * ttt)))
    wmin_esti_cut = wmin_esti[0:length].T

    wzero_esti = np.real(np.fft.ifftshift(np.fft.fft(Fw_amp_esti)))
    M = length // 2
    id = np.argmax(np.abs(wzero_esti))
    wzero_esti_cut = wzero_esti[id - M:id + M + 1].T

    return wmin_esti_cut, wzero_esti_cut


# 求解器模块
def SolverFunc(seis, W, mu1, mu2, D, maxiter, p, tol):
    # p = 2
    # tol = 10e-10
    W=np.array(W)
    D=np.array(D)
    A = np.dot(W.T, W) + mu2 * np.dot(D.T, D)
    inib = np.dot(W.T, seis)
    r = inib
    for k in range(maxiter):
        r1 = r
        Q = mu1 * np.diag(1 / ((abs(r))**(2-p) + np.finfo(float).eps))  # p
        Matrix = A + Q
        G = np.linalg.solve(Matrix, W.T)
        r = np.linalg.solve(Matrix, inib)
        r2 = r
        if np.sum(np.abs(r2 - r1)) / np.sum(np.abs(r1) + np.abs(r2)) < 0.5 * tol:
            break
    return r


# 生成子波
import matplotlib.pyplot as plt
#from reflectivity_modling import *
#from fit_amplitude2min_zero_wavelet2D import *
from scipy.linalg import toeplitz

# Set default figure properties
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['lines.linewidth'] = 2

# Set parameters
N = 180        # Number of reflection coefficient sample points
trace = 30     # Number of model traces

# Generate reflectivity model
modle_name = '1wedge1'
ref = reflectivity_modling(N, modle_name, trace)

plt.imshow(ref)
plt.title('ref')
plt.show()

# Design wavelet
dt = 0.001  # 1ms
fm = 30     # Center frequency of wavelet
trun_time = 0.04
t = np.arange(-trun_time, trun_time + dt, dt)
rowVector = t.reshape((1,-1)) #转为行
w = (1 - 2 * (np.pi * fm * rowVector) ** 2) * np.exp(-(np.pi * fm * rowVector) ** 2)
s=w.T
X=np.zeros((179,1))
d=np.append(s,X,axis=0)
nWaveSampPoint = len(w[0])
W_temp = toeplitz(d, np.zeros(len(ref)))
WW = W_temp[(nWaveSampPoint - 1) // 2 : - (nWaveSampPoint - 1) // 2, :]  # Full freq

plt.plot(WW[:,90])
plt.title('wav')
plt.show()

# 合成地震记录
# Convolution, reverse, shift, multiply, and sum
seis = np.dot(WW, ref)

plt.imshow(seis)
plt.title('seis')
plt.show()

plt.plot(seis[:,15])
plt.title('seis_single')
plt.show()

# 子波提取
# Estimate wavelet  移植部分
L_w = 81
wmin_esti, wzero_esti = fit_amplitude2min_zero_wavelet2D(seis, dt, L_w, 30)
wavelet = wzero_esti / max(wzero_esti)

# Plot original and estimated wavelets
plt.plot(w / max(w), '-k', linewidth=2)
plt.plot(wavelet, '-r', linewidth=2)
plt.title('estimate_wav')
plt.legend(['Original wavelet', 'Estimated wavelet'])
plt.show()


# 求解器求解
mu2 = 0;
mu1 = 0.05;
maxiter = 150;
p = 1;
tol = 10e-20;
D = 0;

r_inv = np.zeros(ref.shape)
for i in range(seis.shape[1]):
    r_inv[:, i] = SolverFunc(seis[:, i], WW, mu1, mu2, D, maxiter, p, tol)
    
plt.imshow(r_inv)
plt.title('solver_ref')
plt.show()


# 求解器求解子波
W_temp_solver = toeplitz(d, np.zeros(len(r_inv)))
WW_solver = W_temp_solver[(nWaveSampPoint - 1) // 2 : - (nWaveSampPoint - 1) // 2, :]  # Full freq

plt.plot(WW_solver[:,90])
plt.title('solver_wav')
plt.show()

#求解器求解合成记录
seis_solver = np.dot(WW_solver, r_inv)

plt.imshow(seis_solver)
plt.title('solver_seis')
plt.show()

plt.plot(seis_solver[:,15])
plt.title('solver_seis_single')
plt.show()
