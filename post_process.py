import numpy as np
import matplotlib.pyplot as plt


DATA_FILE = "Serial-mesh32-convalue0.npz"

data = np.load(DATA_FILE)
train_hist = data['a'][()]

# Divergence
dr = train_hist['delta_real']
dl = train_hist['delta_lose']

plt.plot(range(len(dr)),dr,'r--',range(len(dl)),dl,'bs')
plt.show()


# Contours
x = train_hist['validate'][21] # (1,32,32,3)
print(x.shape)
fig, ax = plt.subplots()
ax.contour(x[0,:,:,0])
plt.show()
