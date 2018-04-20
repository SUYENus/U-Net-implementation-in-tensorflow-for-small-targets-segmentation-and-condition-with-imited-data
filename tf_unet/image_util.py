# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import cv2
from glob import glob
import numpy as np
from PIL import Image
import os

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 3
    n_class = 3
    

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()
            
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        
        train_data, labels = self._post_process(train_data, labels)
        
        nx = train_data.shape[1]
        ny = train_data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),
    
    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels
        
        return label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        return data, labels
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y
    
class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.

    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class = 2):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels = channels

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        return self.data[idx], self.label[idx]


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".jpg", mask_suffix='_mask.jpg', shuffle_data=False, n_class = 2):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        
        self.data_files = self._find_data_files(search_path)
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_data_files(self, search_path):
        all_files = glob(search_path)
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]
    
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
        
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        
        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)
    
        return img,label
def load_data(path='AIO_binary'):
    """
    :Input:  image data size: 480 * 640
    :return: 921600 * 940 Image Data and Mask Data
    """
    w = 480
    h = 640
    os.getcwd()
    os.chdir(path)
    mask_names=glob('*mask.jpg')
    all_names=glob('*.jpg')
    print('found'+str(len(mask_names))+'mask')
    print('found' + str(len(all_names)) + 'files')
    m = len(all_names)
    m_prime = len(mask_names)
    if m != m_prime*2:
        print('image and mask number not matched')
        return
    for i in range(m_prime):
        #mask = np.asarray(Image.open(mask_names[i]))
        #image = np.asarray(Image.open(mask_names[i][:-9]+'.jpg'))
        print('love you love you number',i)
        if i == 0:
            mask_mat = np.reshape(cv2.imread(mask_names[i],0)/255, (w*h*3,1))
            image_mat = np.reshape(cv2.imread(mask_names[i][:-9]+'.jpg'), (w*h*3,1))
        else:
            mask_mat = np.concatenate((mask_mat, np.reshape(cv2.imread(mask_names[i],0)/255, (w*h*3,1))), axis=1)
            image_mat = np.concatenate((image_mat, np.reshape(cv2.imread(mask_names[i][:-9]+'.jpg'), (w*h*3,1))), axis=1)
    np.save('bimage_mat.npy',image_mat)
    np.save('mask_mat.npy',mask_mat)
    return image_mat, mask_mat
def load_bn_data(path='AIO_binary'):
    """
    :Input:  image data size: 480 * 640
    :return: 921600 * 940 Image Data and Mask Data
    """
    w = 480
    h = 640
    os.getcwd()
    os.chdir(path)
    mask_names=glob('*mask.jpg')
    all_names=glob('*.jpg')
    print('found'+str(len(mask_names))+'mask')
    print('found' + str(len(all_names)) + 'files')
    m = len(all_names)
    m_prime = len(mask_names)
    if m != m_prime*2:
        print('image and mask number not matched')
        return
    for i in range(m_prime):
        #mask = np.asarray(Image.open(mask_names[i]))
        #image = np.asarray(Image.open(mask_names[i][:-9]+'.jpg'))
        print('number',i)
        if i == 0:
            mask = cv2.imread(mask_names[i],0)  #mask:480*640*2
            mask[mask!=0] = 1
            mask_2nd = np.zeros((w,h))
            mask_2nd[mask==0] = 1
            mask_2dim = np.concatenate((mask,mask_2nd),axis=2)
            mask_mat = np.reshape(mask_2dim, (w*h*2,1))
            image_mat = np.reshape(cv2.imread(mask_names[i][:-9]+'.jpg')/255, (w*h*3,1))
        else:
            mask = cv2.imread(mask_names[i], 0)
            mask[mask != 0] = 1
            mask = np.reshape(mask,(w, h, 1))
            mask_2nd = np.zeros((w, h, 1))
            mask_2nd[mask == 0] = 1
            mask_2dim = np.concatenate((mask,mask_2nd),axis=2)
            mask_mat = np.concatenate((mask_mat, np.reshape(mask_2dim, (w*h*2,1))), axis=1)
            image_mat = np.concatenate((image_mat, np.reshape(cv2.imread(mask_names[i][:-9]+'.jpg')/255, (w*h*3,1))), axis=1)
    print('saving files')
    np.save('bn_image_mat.npy',image_mat)
    np.save('bn_mask_mat.npy',mask_mat)
    return image_mat, mask_mat

def random_minibatch(train_x, train_y,batch_size,iteration):
    '''
    :param train_x: training samples with type array
    :param train_y: training mask
    :param batch_size
    :return: random mini batch tuple [(x_batch1,y_batch1),(x_batch2,y_batch2)...]
    '''
    num_sample = train_x.shape[1]
    #num_minibatch = batch_size
    permu_index = np.random.permutation(num_sample)
    minibatch_out = []
    #print('preparing minibatches')
    i = iteration + 1

    left = batch_size * (i - 1)
    right = batch_size * i
    print('column'+str(left)+'to'+str(right))
        #batch_index = np.random.permutation(num_sample)[:4]
    #batch_index = permu_index[left:right]
    x_batch =train_x[:,left:right]
    y_batch =train_y[:,left:right]
    minibatch_out.append((x_batch, y_batch))

    #print('Done! Got %d minibatches' %batch_size)
    return minibatch_out





