from tf_unet import unet, util, image_util
import cv2
from matplotlib import pyplot as plt
data_provider = image_util.ImageDataProvider("Data/*.jpg", data_suffix=".jpg", mask_suffix='_mask.jpg',
                                             shuffle_data = True, n_class=3)
net = unet.Unet(layers=3, features_root=64, channels=3, n_class=3)

data, label = data_provider(1)
print(data.shape)
'''
cv2.imshow('label',label[0,...,1])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

prediction = net.predict('3.30/model.cpkt', data)
print(prediction.shape)
cv2.imshow('label',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
pred1 = prediction[0,:,:,:]
#pred2 = prediction[1,:,:,:]
#pred3 = prediction[2,:,:,:]

print(unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape)))

#img = util.combine_img_prediction(data, label, prediction)
#util.save_image(img, "prediction.jpg")





'''
prediction
'''
data, label = data_provider(1)

prediction = net.predict(path, test_x)
mask=prediction[0,:,:,:]
print(label[0,:,:,:])
print(mask)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

img = util.combine_img_prediction(data, label, prediction)
util.save_image(img, "prediction.jpg")