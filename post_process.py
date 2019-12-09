"""
Author:     Richard Tan (rickytan@vt.edu)
Version:    2019.12.08
"""

# ------------------------------------------------------------------------- Imports
import numpy as np, os
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------- Global Variables
DATA_DIR = ".\data"
LABELS = ["lr=0.0005, lambda=0.2", "lr=0.005, lambda=2", "lr=0.005, lambda=0.2"]
LABELS_SINGULARITY = ["removed 9.38% points around singularity", "removed 21.8% points around singularity"]

# ------------------------------------------------------------------------- Divergence

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".npz"):
        filepath = os.path.join(DATA_DIR, filename)
        print("loading %s"%filename)

        # load divergence
        data = np.load(filepath)
        train_hist = data['a'][()]
        dr = train_hist['delta_real']

        # plot divergence
        plt.plot(range(len(dr)),dr)

# show plot
plt.legend(LABELS)
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
