from kan import *
from sklearn.datasets import make_circles, make_moons
import numpy as np

seed = 42
np.random.seed(seed)

fn_map = {
    "cos_pi_x":       lambda x: np.cos(np.pi * x),
    "sin_pi_x":       lambda x: np.sin(np.pi * x),
    "cos_2pi_x":      lambda x: np.cos(2 * np.pi * x),
    "exp_x":          lambda x: np.exp(x),
    "exp_cos_composite": lambda x: np.exp(-np.abs(x)) * np.cos(5 * x**2),
    "abs_x_cos":      lambda x: np.abs(x) * np.cos(2 * np.pi * x),
    "piecewise_smooth": lambda x: np.where(x<0, x**2, np.sqrt(x+1e-5)),
    "x_exp_neg_x2":   lambda x: x * np.exp(-x**2),
    "sin_squared":    lambda x: np.sin(x)**2,
    "sin2_pi_x_squared": lambda x: np.sin(np.pi*x)**2,
    "exp_neg9_x2":    lambda x: np.exp(-9*x**2),
    "cos_squared":    lambda x: np.cos(x)**2,
    "cos_pi_x_pow2":  lambda x: np.cos(np.pi*x)**2,
    "sigmoid_5x":     lambda x: 1/(1+np.exp(-5*x)),
    "tanh":           lambda x: np.tanh(x),
    "tanh_5x":        lambda x: np.tanh(5*x),
    "swish":          lambda x: x/(1+np.exp(-x)),
    "swish_5x":       lambda x: (5*x)/(1+np.exp(-5*x)),
    "heaviside_step": lambda x: np.heaviside(x,0.5),
    "relu":           lambda x: np.maximum(0,x),
    "cubic":          lambda x: x**3,
    "quadratic":      lambda x: x**2 + x,
    "quartic":        lambda x: x**4,
    "quintic":        lambda x: x**5,
    "cubic_poly":     lambda x: x**3 - 3*x,
    "cube_root":      lambda x: np.sign(x)*np.abs(x)**(1/3),
    "sinc_10x":       lambda x: np.where(10*x==0,1.0, np.sin(10*x)/(10*x)),
    "arcsin_5x":      lambda x: np.arcsin(np.clip(5*x,-1,1)),
    "arcsin_pi_x":    lambda x: np.arcsin(np.clip(np.pi*x,-1,1)),
    "arccos_5x":      lambda x: np.arccos(np.clip(5*x,-1,1)),
    "arccos_pi_x":    lambda x: np.arccos(np.clip(np.pi*x,-1,1)),
    "arctan_5x":      lambda x: np.arctan(5*x),
    "arctan_pi_x":    lambda x: np.arctan(np.pi*x),
}
# === Generators (behavior matches your MLP script) ===
def gen_high_freq(n=10000, freq=3):
    X = np.random.rand(n, 2) * 2 - 1
    y = (np.sin(freq * np.pi * X[:, 0]) > 0).astype(int)
    return X, y

def gen_high_freq2(n=10000, freq=4):
    X = np.random.rand(n, 2) * 2 - 1
    y = (np.sin(freq * np.pi * X[:, 0]) > 0).astype(int)
    return X, y

def gen_multibit_xor3(n=10000, bits=3):
    X = np.random.randint(0, 2, size=(n, bits))
    y = np.bitwise_xor.reduce(X, axis=1)
    return X.astype(float), y

def gen_multi_rings(n=10000, k=6):
    theta = np.random.uniform(0, 2*np.pi, n)
    r = np.random.uniform(0.2, 1.0, n)
    X = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    y = ((r*k) % 2 > 1).astype(int)
    return X, y

def gen_polar_checker(n=10000, r_freq=5, theta_freq=5):
    r = np.random.uniform(0.1, 1.0, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    y_class = (((np.floor(r*r_freq) + np.floor(theta/(2*np.pi)*theta_freq)) % 2) > 0).astype(int)
    return np.column_stack([x, y]), y_class

def gen_rotated_bar(n=10000, angle=np.pi/7, width=0.1):
    X = np.random.uniform(-1,1,(n,2))
    x_rot = X[:,0]*np.cos(angle) + X[:,1]*np.sin(angle)
    y = (np.abs(x_rot) < width).astype(int)
    return X, y

def gen_two_arm_spiral(n=10000, noise=0.2):
    n2 = n//2
    t = np.linspace(0, np.pi, n2)
    x1 = t*np.cos(t) + np.random.randn(n2)*noise
    y1 = t*np.sin(t) + np.random.randn(n2)*noise
    x2 = -t*np.cos(t) + np.random.randn(n2)*noise
    y2 = -t*np.sin(t) + np.random.randn(n2)*noise
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.array([0]*n2 + [1]*n2)
    return X, y

def gen_spirals(n=10000, noise=0.2):
    return gen_two_arm_spiral(n, noise)

def gen_gaussian_cross(n=10000, sigma_bar=0.05, sigma_blob=0.1, bar_limit=0.9, blob_offset=0.5):
    cross_n = n//2
    h_n = cross_n//2
    v_n = cross_n - h_n
    xh = np.random.uniform(-bar_limit, bar_limit, h_n)
    yh = np.random.normal(0, sigma_bar, h_n)
    xv = np.random.normal(0, sigma_bar, v_n)
    yv = np.random.uniform(-bar_limit, bar_limit, v_n)
    Xc = np.vstack([np.column_stack([xh,yh]), np.column_stack([xv,yv])])
    yc = np.ones(len(Xc), dtype=int)
    blob_n = n-cross_n
    per_corner = blob_n//4
    rem = blob_n - 4*per_corner
    corners = [(-blob_offset,-blob_offset),(-blob_offset,blob_offset),(blob_offset,-blob_offset),(blob_offset,blob_offset)]
    Xb, yb = [], []
    for i,(cx,cy) in enumerate(corners):
        cnt = per_corner + (1 if i<rem else 0)
        xb = np.random.normal(cx, sigma_blob, cnt)
        yb_chunk = np.random.normal(cy, sigma_blob, cnt)
        Xb.append(np.column_stack([xb,yb_chunk])); yb.extend([0]*cnt)
    Xb = np.vstack(Xb); yb = np.array(yb, dtype=int)
    X = np.vstack([Xc, Xb]); y = np.hstack([yc, yb])
    return X, y

# --- A-little-harder wrappers ---
def gen_high_freq_alittle1(n=10000, freq=4): return gen_high_freq(n, freq)
def gen_high_freq_alittle2(n=10000, freq=5): return gen_high_freq(n, freq)
def gen_high_freq_alittle3(n=10000, freq=6): return gen_high_freq(n, freq)
def gen_multibit_xor_alittle1(n=10000, bits=3): return gen_multibit_xor3(n, bits)
def gen_multibit_xor_alittle2(n=10000, bits=4): return gen_multibit_xor3(n, bits)
def gen_multibit_xor_alittle3(n=10000, bits=5): return gen_multibit_xor3(n, bits)
def gen_multi_rings_alittle1(n=10000, k=4): return gen_multi_rings(n, k)
def gen_multi_rings_alittle2(n=10000, k=5): return gen_multi_rings(n, k)
def gen_multi_rings_alittle3(n=10000, k=6): return gen_multi_rings(n, k)
def gen_polar_checker_alittle1(n=10000, r_freq=3, theta_freq=3): return gen_polar_checker(n, r_freq, theta_freq)
def gen_polar_checker_alittle2(n=10000, r_freq=4, theta_freq=4): return gen_polar_checker(n, r_freq, theta_freq)
def gen_polar_checker_alittle3(n=10000, r_freq=5, theta_freq=5): return gen_polar_checker(n, r_freq, theta_freq)
def gen_rotated_bar_alittle1(n=10000, angle=np.pi/8, width=0.2): return gen_rotated_bar(n, angle, width)
def gen_rotated_bar_alittle2(n=10000, angle=np.pi/6, width=0.15): return gen_rotated_bar(n, angle, width)
def gen_rotated_bar_alittle3(n=10000, angle=np.pi/4, width=0.1): return gen_rotated_bar(n, angle, width)
def gen_two_arm_spiral_alittle1(n=10000, noise=0.1): return gen_two_arm_spiral(n, noise)
def gen_two_arm_spiral_alittle2(n=10000, noise=0.15): return gen_two_arm_spiral(n, noise)
def gen_two_arm_spiral_alittle3(n=10000, noise=0.2): return gen_two_arm_spiral(n, noise)
def gen_spirals_alittle1(n=10000, noise=0.1): return gen_spirals(n, noise)
def gen_spirals_alittle2(n=10000, noise=0.15): return gen_spirals(n, noise)
def gen_spirals_alittle3(n=10000, noise=0.2): return gen_spirals(n, noise)
def gen_gaussian_cross_alittle1(n=10000, sigma_bar=0.1, sigma_blob=0.2, bar_limit=0.9, blob_offset=0.4): return gen_gaussian_cross(n, sigma_bar, sigma_blob, bar_limit, blob_offset)
def gen_gaussian_cross_alittle2(n=10000, sigma_bar=0.08, sigma_blob=0.15, bar_limit=0.8, blob_offset=0.5): return gen_gaussian_cross(n, sigma_bar, sigma_blob, bar_limit, blob_offset)
def gen_gaussian_cross_alittle3(n=10000, sigma_bar=0.05, sigma_blob=0.1, bar_limit=0.7, blob_offset=0.6): return gen_gaussian_cross(n, sigma_bar, sigma_blob, bar_limit, blob_offset)
def gen_checkerboard_alittle1(n=10000, grid=3):
    X = np.random.rand(n,2)
    y = ((np.floor(X[:,0]*grid)+np.floor(X[:,1]*grid))%2>0).astype(int)
    return X*2-1, y
def gen_checkerboard_alittle2(n=10000, grid=4): return gen_checkerboard_alittle1(n, grid)
def gen_checkerboard_alittle3(n=10000, grid=5): return gen_checkerboard_alittle1(n, grid)
def gen_ellipse_alittle1(n=10000, a=0.7, b=0.5):
    X = np.random.uniform(-1,1,(n,2))
    y = ((X[:,0]/a)**2+(X[:,1]/b)**2<=1).astype(int)
    return X, y
def gen_ellipse_alittle2(n=10000, a=0.6, b=0.4): return gen_ellipse_alittle1(n, a, b)
def gen_ellipse_alittle3(n=10000, a=0.5, b=0.3): return gen_ellipse_alittle1(n, a, b)
def HardBrexit(n=10000): return gen_gaussian_cross_alittle1(n)

# === Base sklearn tasks ===
def task_moons(n=10000, noise=0.1, random_state=seed):
    X, y = make_moons(n_samples=n, noise=noise, random_state=random_state)
    return X.astype(float), y.astype(int)

def task_circles(n=10000, noise=0.08, factor=0.5, random_state=seed):
    X, y = make_circles(n_samples=n, noise=noise, factor=factor, random_state=random_state)
    return X.astype(float), y.astype(int)
TASKS = [
    ("moons", lambda: task_moons()),
    ("circles", lambda: task_circles()),
    ("gaussian_cross", lambda: gen_gaussian_cross()),
    ("high_freq2", lambda: gen_high_freq2()),
    ("high_freq", lambda: gen_high_freq()),
    ("multi_rings", lambda: gen_multi_rings()),
    ("polar_checker", lambda: gen_polar_checker()),
    ("rotated_bar", lambda: gen_rotated_bar()),
    ("two_arm_spiral", lambda: gen_two_arm_spiral()),
    ("spirals", lambda: gen_spirals()),
    ("high_freq_a1", lambda: gen_high_freq_alittle1()),
    ("high_freq_a2", lambda: gen_high_freq_alittle2()),
    ("high_freq_a3", lambda: gen_high_freq_alittle3()),
    ("rings_a1", lambda: gen_multi_rings_alittle1()),
    ("rings_a2", lambda: gen_multi_rings_alittle2()),
    ("rings_a3", lambda: gen_multi_rings_alittle3()),
    ("polar_a1", lambda: gen_polar_checker_alittle1()),
    ("polar_a2", lambda: gen_polar_checker_alittle2()),
    ("polar_a3", lambda: gen_polar_checker_alittle3()),
    ("bar_a1", lambda: gen_rotated_bar_alittle1()),
    ("bar_a2", lambda: gen_rotated_bar_alittle2()),
    ("bar_a3", lambda: gen_rotated_bar_alittle3()),
    ("gauss_a1", lambda: gen_gaussian_cross_alittle1()),
    ("gauss_a2", lambda: gen_gaussian_cross_alittle2()),
    ("gauss_a3", lambda: gen_gaussian_cross_alittle3()),
    ("checker_a1", lambda: gen_checkerboard_alittle1()),
    ("checker_a2", lambda: gen_checkerboard_alittle2()),
    ("checker_a3", lambda: gen_checkerboard_alittle3()),
    ("ellipse_a1", lambda: gen_ellipse_alittle1()),
    ("ellipse_a2", lambda: gen_ellipse_alittle2()),
    ("ellipse_a3", lambda: gen_ellipse_alittle3()),
    ("HardBrexit", lambda: HardBrexit()),
]