import numpy as np
import matplotlib.pyplot as plt
from grad import gradient

func = 'sin'
plt.plot(gradient(func=func,x=np.linspace(-100,100,1000)))
plt.savefig(func+'_gradient.png')
plt.show()