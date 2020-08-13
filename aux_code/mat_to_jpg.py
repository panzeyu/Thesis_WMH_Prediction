# This script should be placed under the main folder
#  i.e., the JMW_Data_mining_brain_Res folder


import numpy as np
from scipy.io import loadmat
import cv2


i = 1
for _ in range(1):
    # looping over DMP18-DMP60 as the training dataset
    # Change to [1, 18] to get the validation dataset
    # Switch between V1/V3 for baseline and follow-up images
    for j in range(18, 60):
        path_t1 = 'DMP' + str(j) + '/V3/t1.mat'
        path_t2 = 'DMP' + str(j) + '/V3/t2.mat'
        path_flair = 'DMP' + str(j) + '/V3/flair.mat'

        f1 = loadmat(path_t1)
        f2 = loadmat(path_t2)
        f3 = loadmat(path_flair)

        t1 = f1['t1']
        t2 = f2['t2']
        flair = f3['flair']

        # Looping over slices within volume
        # Change the indices as preferred
        for k in range(11,40):
            t1_1 = t1[:, :, k]
            t2_1 = t2[:, :, k]
            flair_1 = flair[:, :, k]

            # Stacking 3 channels
            mat = np.zeros((256, 256, 3))
            mat[:, :, 0] = t1_1
            mat[:, :, 1] = t2_1
            mat[:, :, 2] = flair_1

            # Pixel normalization & encoding
            img = cv2.normalize(src=mat, dst=None, alpha=0, beta=255,
                                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            name = str(i) + '.jpg'
            # Save to whatever path
            cv2.imwrite('/Users/zeyupan/Desktop/float_32/' + name, img)
            i += 1





