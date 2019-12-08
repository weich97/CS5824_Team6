"""
Author:     Richard Tan (rickytan@vt.edu)
Version:    2019.12.08
"""

# ------------------------------------------------------------------------- Imports
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------- Global Variables
DATA_FILE = ".\data\lr_0005__lambda_02.npz"

# load the specified file
data = np.load(DATA_FILE)
train_hist = data['a'][()]

# ------------------------------------------------------------------------- Divergence
dr = train_hist['delta_real']
dl = train_hist['delta_lose']

# plot divergence
plt.plot(range(len(dr)),dr,'r--',range(len(dl)),dl,'bs')
plt.yscale('log')
plt.ylabel('mean divergence')
plt.xlabel('epoch')
plt.show()


# ------------------------------------------------------------------------- Contours
# x = train_hist['validate'][0] # (1,32,32,3)
# print(x.shape)
# fig, ax = plt.subplots()
# ax.contour(x[0,:,:,0])
# plt.show()
