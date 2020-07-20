import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read file
cv_file = cv2.FileStorage('bof_db.xml', cv2.FILE_STORAGE_READ)
bof_list = cv_file.getNode('BOF_LIST').mat()

# plot with correlations
plt.style.use('ggplot')
fig, ax = plt.subplots(6, 4, figsize=(9,7))
x = np.linspace(0, 1, 180)
bof_src = bof_list[0, :]
j, k = 0, 0
for i in range(len(bof_list)):
    bof_tar = bof_list[i, :]
    corr = np.corrcoef(bof_src, bof_tar)[0, 1]
    if k > 3:
        k = 0
        j += 1
    ax[j,k].plot(x, bof_src, x, bof_tar)
    ax[j,k].set_title(r'$R_{x,y}=%.2f$' % corr, {'fontsize': 11})
    k += 1
plt.tight_layout()
plt.show()

# close file
cv_file.release()
