import numpy as np
from PIL import Image
from his import *
from data_set import *
from util import *
from his_match import *
import matplotlib.pyplot as plt
def get_his_feature(source, reference, mask_src, mask_ref):
    source_feature = mask_src * source
    reference_feature = mask_ref * reference
    src_temp = np.copy(source_feature)
    mixed_feature = histogram_matching(src_temp, reference_feature)
    return mixed_feature

makeup_img = np.array(Image.open('./all/images/makeup/vHX581.png'))
makeup_img = get_transform(makeup_img)
makeup_img = np.reshape(makeup_img, (1,256,256,3))


makeup_seg = np.array(Image.open('./all/segs/makeup/vHX581.png'))
makeup_seg = makeup_seg[:,:,np.newaxis]
makeup_seg = np.tile(makeup_seg,3)
makeup_seg = get_feature(makeup_seg, [feature.Ulip.value, feature.Dlip.value])
makeup_seg = get_transform(makeup_seg)
makeup_seg[makeup_seg >= 0.5] = 1
makeup_seg[makeup_seg < 0.5] = 0
makeup_seg = np.reshape(makeup_seg, (1,256,256,3))



Nmakeup_img = np.array(Image.open('./all/images/non-makeup/vSYYZ2.png'))
Nmakeup_img = get_transform(Nmakeup_img)
Nmakeup_img = np.reshape(Nmakeup_img, (1,256,256,3))

Nmakeup_seg = np.array(Image.open('./all/segs/non-makeup/vSYYZ2.png'))
Nmakeup_seg = Nmakeup_seg[:,:,np.newaxis]
Nmakeup_seg = np.tile(Nmakeup_seg,3)
Nmakeup_seg = get_feature(Nmakeup_seg, [feature.Ulip.value, feature.Dlip.value])
Nmakeup_seg = get_transform(Nmakeup_seg)
Nmakeup_seg[Nmakeup_seg >= 0.5] = 1
Nmakeup_seg[Nmakeup_seg < 0.5] = 0
Nmakeup_seg = np.reshape(Nmakeup_seg, (1,256,256,3))


# makeup_seg
makeup_lip = makeup_img * makeup_seg
makeup_lip = makeup_lip.astype(np.uint8)
Nmakeup_lip = Nmakeup_img * Nmakeup_seg
Nmakeup_lip = Nmakeup_lip.astype(np.uint8)
# makeup_lip_rgb = makeup_lip[makeup_lip > 0]
# makeup_lip_rgb = np.reshape(makeup_lip_rgb, (makeup_lip_rgb.size//3,3))
# Nmakeup_lip_rgb = makeup_lip[Nmakeup_lip > 0]
# Nmakeup_lip_rgb = np.reshape(Nmakeup_lip_rgb, (Nmakeup_lip_rgb.size//3,3))
# makeup_lip[makeup_lip == 0] = 100000
# Nmakeup_lip[Nmakeup_lip == 0] = 100000
# print(makeup_lip_rgb.shape)
feature = hist_match(Nmakeup_lip[0], makeup_lip[0], 3)
# img = Nmakeup_img[0] - Nmakeup_lip[0] + feature
# print(np.max(makeup_lip))
plt.imsave('./lip/makeup-3', makeup_lip[0])
plt.imsave('./lip/Nmakeup-3', Nmakeup_lip[0])
plt.imsave('./lip/his-3', feature)
