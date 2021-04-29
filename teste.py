import numpy as np
from matplotlib import pyplot as plt

t = np.array([0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9])
x = 4* np.sin(2 * np.pi * t)

print(x)
plt.plot (t, x)
plt.show()