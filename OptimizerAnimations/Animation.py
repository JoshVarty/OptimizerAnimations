#http://tiao.io/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
import matplotlib.pyplot as plt
import autograd.numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad, grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

f  = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
xmin, xmax, xstep = -4.5, 4.5, .01
ymin, ymax, ystep = -4.5, 4.5, .01

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x, y)


minima = np.array([3., .5])
minima_ = minima.reshape(-1, 1)

x0 = np.array([1., 1.5])

def make_minimize_cb(path=[]):
    
    def minimize_cb(xk):
        # note that we make a deep copy of xk
        path.append(np.copy(xk))

    return minimize_cb


class TrajectoryAnimation(animation.FuncAnimation):
    
    def __init__(self, *paths, labels=[], fig=None, ax=None, frames=None, interval=60, repeat_delay=5, blit=True, **kwargs):

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax
        
        self.paths = paths

        if frames is None:
            frames = max(path.shape[1] for path in paths)
  
        self.lines = [ax.plot([], [], label=label, lw=2)[0] 
                      for _, label in zip_longest(paths, labels)]
        self.points = [ax.plot([], [], 'o', color=line.get_color())[0] 
                       for line in self.lines]

        super(TrajectoryAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                  frames=frames, interval=interval, blit=blit,
                                                  repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            point.set_data([], [])

        x = self.lines + self.points
        return x

    def animate(self, i):
        for line, point, path in zip(self.lines, self.points, self.paths):
            line.set_data(*path[::,:i])
            point.set_data(*path[::,i-1:i])

        x = self.lines + self.points
        return x


methods = [
    "SGD",
    "Momentum",
    "Nesterov",
    "AdaGrad",
    "RMSProp",
    "Adam"
]

def stochasticGradientDescent(function, x0, y0, learning_rate, num_steps):

    allX = [x0]
    allY = [y0]

    x = x0
    y = y0

    for _ in range(num_steps):
        
        dz_dx = grad(function, argnum=0)(x, y)
        dz_dy = grad(function, argnum=1)(x, y)

        x = x - dz_dx * learning_rate
        y = y - dz_dy * learning_rate

        allX.append(x)
        allY.append(y)

    return np.array([allX, allY])

def momentumUpdate(function, x0, y0, learning_rate, num_steps, momentum = 0.9):

    allX = [x0]
    allY = [y0]

    x = x0
    y = y0
    
    x_v = 0
    y_v = 0

    for _ in range(num_steps):
        
        dz_dx = grad(function, argnum=0)(x, y)
        dz_dy = grad(function, argnum=1)(x, y)
        
        x_v = (momentum * x_v) - (dz_dx * learning_rate)
        y_v = (momentum * y_v) - (dz_dy * learning_rate)

        x = x + x_v
        y = y + y_v

        allX.append(x)
        allY.append(y)

    return np.array([allX, allY])

def nesterovMomentumUpdate(function, x0, y0, learning_rate, num_steps, momentum = 0.9):
    allX = [x0]
    allY = [y0]

    x = x0
    y = y0
    
    x_v= 0
    x_v_prev = 0
    y_v = 0
    y_v_prev = 0

    for _ in range(num_steps):
        
        dz_dx = grad(function, argnum=0)(x, y)
        dz_dy = grad(function, argnum=1)(x, y)

        x_v_prev = x_v
        x_v = (momentum * x_v) - (dz_dx * learning_rate)
        x = x  - momentum * x_v_prev + (1 + momentum) * x_v

        y_v_prev = y_v
        y_v = (momentum * y_v) - (dz_dy * learning_rate)
        y = y - momentum * y_v_prev + (1 + momentum) * y_v

        allX.append(x)
        allY.append(y)

    return np.array([allX, allY])

def adaGradUpdate(function, x0, y0, learning_rate, num_steps):
    allX = [x0]
    allY = [y0]

    x = x0
    y = y0

    x_cache = 0
    y_cache = 0

    for _ in range(num_steps):

        dz_dx = grad(function, argnum=0)(x, y)
        dz_dy = grad(function, argnum=1)(x, y)

        x_cache = x_cache + dz_dx ** 2
        x = x - learning_rate * dz_dx / (np.sqrt(x_cache) + 1e-7)
        
        y_cache = y_cache + dz_dy ** 2
        y = y - learning_rate * dz_dy / (np.sqrt(y_cache) + 1e-7)

        allX.append(x)
        allY.append(y)

    return np.array([allX, allY])

def rmsPropUpdate(function, x0, y0, learning_rate, num_steps, decay_rate = 0.9):
    allX = [x0]
    allY = [y0]

    x = x0
    y = y0

    x_cache = 0
    y_cache = 0

    for _ in range(num_steps):

        dz_dx = grad(function, argnum=0)(x, y)
        dz_dy = grad(function, argnum=1)(x, y)

        x_cache = decay_rate * x_cache + (1 - decay_rate) * dz_dx**2
        x = x - learning_rate * dz_dx / (np.sqrt(x_cache) + 1e-7)
        
        y_cache = decay_rate * y_cache + (1 - decay_rate) * dz_dy**2
        y = y - learning_rate * dz_dy / (np.sqrt(y_cache) + 1e-7)

        allX.append(x)
        allY.append(y)

    return np.array([allX, allY])


def adamUpdate(function, x0, y0, learning_rate, num_steps, beta1, beta2):
    allX = [x0]
    allY = [y0]

    x = x0
    y = y0

    m_x = 0
    m_y = 0
    x_v = 0
    y_v = 0

    #We need 1-indexed steps
    for step in range(1, num_steps + 1):
        dz_dx = grad(function, argnum=0)(x, y)
        dz_dy = grad(function, argnum=1)(x, y)

        m_x = beta1 * m_x + (1 - beta1)  * dz_dx
        m_step_x = m_x / (1-beta1**step)
        x_v = beta2 * x_v + (1 - beta2) * (dz_dx ** 2)
        v_step_x = x_v / (1 - beta2 ** step)
        x = x - learning_rate * m_step_x / (np.sqrt(v_step_x) + 1e-7)
        
        m_y = beta1 * m_y + (1 - beta1)  * dz_dy
        m_step_y = m_y / (1-beta1**step)
        y_v = beta2 * y_v + (1 - beta2) * (dz_dy ** 2)
        v_step_y = y_v / (1 - beta2 ** step)
        y = y - learning_rate * m_step_y / (np.sqrt(v_step_y) + 1e-7)

        allX.append(x)
        allY.append(y)

    return np.array([allX, allY])

learning_rate = 0.005
num_steps = 100
sgdPath = stochasticGradientDescent(f, x0[0], x0[1], learning_rate, num_steps)
momentumPath =  momentumUpdate(f, x0[0], x0[1], learning_rate, num_steps)
nesterovPath = nesterovMomentumUpdate(f, x0[0], x0[1], learning_rate, num_steps)

#These ones need different learning rates or they perform poorly...
adaGrad = adaGradUpdate(f, x0[0], x0[1], 0.5, num_steps)
rmsProp = rmsPropUpdate(f, x0[0], x0[1], 0.05, num_steps)
adam = adamUpdate(f, x0[0], x0[1], 0.1, num_steps, 0.9, 0.999)

paths = [sgdPath, momentumPath, nesterovPath, adaGrad, rmsProp, adam]

fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
ax.plot(*minima_, 'r*', markersize=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

anim = TrajectoryAnimation(*paths, labels=methods, ax=ax)

ax.legend(loc='upper left')

html = "<html><body>" + anim.to_html5_video() + "</body></html>"
file = open('animation.html', 'w')
file.write(html)
file.close()