
'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

from tf_unet import util
from tf_unet.layers import (weight_variable, weight_variable_devonc, bias_variable, 
                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
                            cross_entropy, recall, xavier_weights)
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def get_image_mask():
    '''

    :return: read image and mask from directories and return image queue and mask queue for batch operation
    '''
    image_names = tf.train.match_filenames_once('/home/simon/deeplearning/models/UNET/image/*.jpg')
    imgname_queue = tf.train.string_input_producer(image_names, shuffle=False)
    reader = tf.WholeFileReader()
    key, img_value = reader.read(imgname_queue)
    images = tf.image.decode_jpeg(img_value, channels=3)
    images.set_shape((600, 800, 3))

    mask_names = tf.train.match_filenames_once('/home/simon/deeplearning/models/UNET/mask/*.jpg')
    maskname_queue = tf.train.string_input_producer(mask_names, shuffle=False)
    reader = tf.WholeFileReader()
    key, mask_value = reader.read(maskname_queue)
    mask = tf.image.decode_jpeg(mask_value, channels=3)
    mask.set_shape((600, 800, 3))

    return images, mask

def batch_op(images, mask, batch_size, shuffle=True):
    '''

    :param images: image queue
    :param mask: mask queue
    :param batch_size:
    :return: batch operation for images and mask
    '''
    if shuffle is True:
        num_preprocess_threads = 1
        min_queue_examples = 200
        image_op, mask_op = tf.train.shuffle_batch(
                            [images, mask],
                            batch_size=batch_size,
                            num_threads=num_preprocess_threads,
                            capacity=min_queue_examples + 3 * batch_size,
                            min_after_dequeue=min_queue_examples)
        zero = tf.zeros((batch_size,600,800,3))
        one = tf.ones((batch_size,600,800,3))
        mask_op = tf.where(mask_op>128, one, zero)
        return image_op, mask_op
    else:
        num_preprocess_threads = 1
        min_queue_examples = 200
        image_op, mask_op = tf.train.batch(
            [images, mask],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
        zero = tf.zeros((batch_size, 600, 800, 3))
        one = tf.ones((batch_size, 600, 800, 3))
        mask_op = tf.where(mask_op > 128, one, zero)
        return image_op, mask_op



def create_conv_net(x, mode, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True,norm = True):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of 3.30 labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]
 
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    
    in_size = 1000
    size = in_size
    if norm:
        # down layers
        for layer in range(0, layers):
            features = 2**layer*features_root
            if layer == 0:
                #w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
                with tf.variable_scope('conv1_layer{}'.format(layer)):
                    w1 = xavier_weights('W1_conv{}'.format(layer),[filter_size,filter_size,channels,features])
            else:
                with tf.variable_scope('conv1_layer{}'.format(layer)):
                    w1 = xavier_weights('W1_conv{}'.format(layer),[filter_size, filter_size, features//2, features])
            with tf.variable_scope('conv2_layer{}'.format(layer)):
                w2 = xavier_weights('W2_conv{}'.format(layer),[filter_size, filter_size, features, features])

            conv1 = conv2d(in_node, w1, keep_prob)
            bn_conv1 = tf.layers.batch_normalization(
                conv1, training=mode, name = 'bn_layer{}_1'.format(layer)
            )
            tmp_h_conv = tf.nn.relu(bn_conv1)
            conv2 = conv2d(tmp_h_conv, w2, keep_prob)
            bn_conv2 = tf.layers.batch_normalization(
                conv2, training=mode, name = 'bn_layer{}_2'.format(layer)
            )
            dw_h_convs[layer] = tf.nn.relu(bn_conv2)

            weights.append((w1, w2))
            convs.append((conv1, conv2))

            size -= 4
            if layer < layers-1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

        in_node = dw_h_convs[layers-1]

        # up layers
        for layer in range(layers-2, -1, -1):
            features = 2**(layer+1)*features_root
            with tf.variable_scope('dconv_layer{}'.format(layer)):
                wd = weight_variable_devonc('W_dec{}'.format(layer),[pool_size, pool_size, features//2, features])
                bd = bias_variable('b_dec{}'.format(layer),[features//2])
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat
            with tf.variable_scope('dconv2_layer{}'.format(layer)):
                w1 = xavier_weights('W1_dec_{}'.format(layers), [filter_size, filter_size, features, features//2])
                w2 = xavier_weights('W2_dec_{}'.format(layers), [filter_size, filter_size, features//2, features//2])
                b1 = bias_variable('b1_dec_{}'.format(layers), [features//2])
                b2 = bias_variable('b2_dec_{}'.format(layers), [features//2])

            conv1 = conv2d(h_deconv_concat, w1, keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(h_conv, w2, keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))


            size *= 2
            size -= 4

        # Output Map
        weight = xavier_weights('out_weights',[1, 1, features_root, n_class])
        bias = bias_variable('out_bias',[n_class])
        conv = conv2d(in_node, weight, tf.constant(1.0))
        output_map = conv + bias
        up_h_convs["out"] = output_map

        if summaries:
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01'%i, get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02'%i, get_image_summary(c2))

            for k in pools.keys():
                tf.summary.image('summary_pool_%02d'%k, get_image_summary(pools[k]))

            for k in deconv.keys():
                tf.summary.image('summary_deconv_concat_%02d'%k, get_image_summary(deconv[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k])

        variables = []
        for w1,w2 in weights:
            variables.append(w1)
            variables.append(w2)

        for b1,b2 in biases:
            variables.append(b1)
            variables.append(b2)
    else:
        # down layers
        for layer in range(0, layers):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
            else:
                w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev)

            w2 = weight_variable([filter_size, filter_size, features, features], stddev)
            with tf.variable_scope('upsampling_layer{}'.format(layer)):
                b1 = bias_variable('conv_b1',[features])
                b2 = bias_variable('conv_b2',[features])

            conv1 = conv2d(in_node, w1, keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(tmp_h_conv, w2, keep_prob)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 4
            if layer < layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

        in_node = dw_h_convs[layers - 1]

        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            with tf.variable_scope('upsampling_layer{}'.format(layer)):
                wd = weight_variable_devonc('deconv_wd_layer{}'.format(layer),[pool_size, pool_size, features // 2, features])
                bd = bias_variable('deconv_bd',[features // 2])
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat
            with tf.variable_scope('upsampling_layer{}'.format(layer)):
                w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev)
                w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev)
                b1 = bias_variable('deconv_b1',[features // 2])
                b2 = bias_variable('deconv_b2',[features // 2])

            conv1 = conv2d(h_deconv_concat, w1, keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(h_conv, w2, keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= 2
            size -= 4

        # Output Map
        weight = weight_variable([1, 1, features_root, n_class], stddev)
        bias = bias_variable('out_bias',[n_class])
        conv = conv2d(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs["out"] = output_map

        if summaries:
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01' % i, get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i, get_image_summary(c2))

            for k in pools.keys():
                tf.summary.image('summary_pool_%02d' % k, get_image_summary(pools[k]))

            for k in deconv.keys():
                tf.summary.image('summary_deconv_concat_%02d' % k, get_image_summary(deconv[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

        variables = []
        for w1, w2 in weights:
            variables.append(w1)
            variables.append(w2)

        for b1, b2 in biases:
            variables.append(b1)
            variables.append(b2)

    return output_map, variables, int(in_size - size)


class Unet(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of 3.30 label
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """
    
    def __init__(self, channels=3, n_class=2, cost="cross_entropy", norm=True, cost_kwargs={}, **kwargs):
        tf.reset_default_graph()
        self.norm = norm
        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)
        
        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        self.mode = tf.placeholder(tf.bool, name='mode')

        logits, self.variables, self.offset = create_conv_net(self.x, self.mode, self.keep_prob, channels, n_class, norm = self.norm, **kwargs)
        
        self.cost = self._get_cost(logits, cost, cost_kwargs)
        
        self.gradients_node = tf.gradients(self.cost, self.variables)
         
        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                                          tf.reshape(pixel_wise_softmax_2(logits), [-1, n_class])))
        self.predicter = pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        #self.pd = tf.placeholder('float', shape=[None,None,None,n_class])
        self.recall_c0, self.recall_c1 = recall(self.predicter, self.y)

    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are: 
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """
        
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels_hotcode = tf.reshape(self.y, [-1, self.n_class])
        labels = tf.argmax(self.y, axis=3)
        flat_labels = tf.reshape(labels, [-1])

        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)
            
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
                weight_map = tf.multiply(flat_labels_hotcode, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)
        
                loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                   labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)
        
                loss = tf.reduce_mean(weighted_loss)
                
            else:
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                              labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax_2(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union =  eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection/ (union))
            
        else:
            raise ValueError("Unknown cost function: "%cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += (regularizer * regularizers)
            
        return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        #init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            #sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1., self.mode:True})
            
        return prediction
    
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    
    """

    
    def __init__(self, net, learning_rate, batch_size=1, norm_grads=False, optimizer="momentum", verify_size = 2, n_class=2, opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.n_class = n_class
        self.verification_batch_size = verify_size
        self.lr = learning_rate
    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learn_rate')
            #learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost, 
                                                                                global_step=global_step)
        elif self.optimizer == "adam":
            #learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learn_rate')
            self.learning_rate_node = self.learning_rate# tf.Variable(self.learning_rate)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                     global_step=global_step)
        
        return optimizer
        
    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)
        
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))
        
        if self.net.summaries and self.norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        
        return init

    def train(self, output_path, training_iters=301, epochs=100, dropout=0.75, step=1, SGD = False, restore=False, write_graph=False, prediction_path = 'prediction'):
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the 3.30 path
        :param prediction_path: path where to save predictions on each epoch
        """


        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        
        init = self._initialize(training_iters, output_path, restore, prediction_path)
        image, mask = get_image_mask()
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)


            x_train_op, y_train_op = batch_op(image, mask, self.batch_size)
            x_test_op, y_test_op = batch_op(image, mask, self.verification_batch_size)
            #load test data
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            x_test, y_test = sess.run([x_test_op, y_test_op])


            pred_shape = self.store_prediction(sess, x_test, y_test, "_init")
            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")
            
            avg_gradients = None

            if not SGD:
                print('using BG')
                for epoch in range(epochs):
                    train_x = np.reshape(train_x,[-1,600,800,3])
                    train_y = np.reshape(train_y,[-1,600,800,self.n_class])
                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                        feed_dict={self.net.x: train_x,
                                   self.net.y: util.crop_to_shape(train_y, pred_shape),
                                   self.net.keep_prob: dropout,
                                   self.net.mode:True})

                    if self.net.summaries and self.norm_grads:
                        avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                        norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                        self.norm_gradients_node.assign(norm_gradients).eval()

                    self.output_minibatch_stats(sess, summary_writer, step, train_x,
                                                util.crop_to_shape(train_y, pred_shape))
                    step += 1
                    self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                    self.store_prediction(sess, x_test, y_test, "epoch_%s" % epoch)
                    save_path = self.net.save(sess, save_path)

                logging.info("Optimization Finished!")

                return save_path

            else:
                for epoch in range(epochs):
                    total_loss = 0
                    step = 1
                    for i in range(training_iters):
                        batch_x, batch_y = sess.run([x_train_op, y_train_op])

                        # Run optimization op (backprop)
                        _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate_node),
                                                          feed_dict={self.net.x: batch_x,
                                                                     self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                                     self.net.keep_prob: dropout,
                                                                     self.net.mode: True,
                                                                     self.learning_rate: self.lr})

                        if self.net.summaries and self.norm_grads:
                            avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                            norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                            self.norm_gradients_node.assign(norm_gradients).eval()
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x,
                                                        util.crop_to_shape(batch_y, pred_shape),loss)
                        total_loss += loss
                        step += 1

                    self.output_epoch_stats(epoch, total_loss, training_iters, lr)

                    self.store_prediction(sess, x_test, y_test, "epoch_%s"%epoch)

                    save_path = self.net.save(sess, save_path)
                coord.request_stop()
                logging.info("Optimization Finished!")

                return save_path
        
    def store_prediction(self, sess, batch_x, batch_y, name):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x, 
                                                             self.net.y: batch_y, 
                                                             self.net.keep_prob: 1.,
                                                             self.net.mode: False,
                                                             self.learning_rate: self.lr})
        pred_shape = prediction.shape
        #print(pred_shape)
        loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x, 
                                                       self.net.y: util.crop_to_shape(batch_y, pred_shape), 
                                                       self.net.keep_prob: 1.,
                                                       self.net.mode: False,
                                                       self.learning_rate: self.lr})
        recall_c0, recall_c1 = sess.run((self.net.recall_c0,self.net.recall_c1),feed_dict={self.net.x: batch_x,
                                                       self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                       self.net.keep_prob: 1.,
                                                       self.net.mode: False,
                                                       self.learning_rate: self.lr})#,
                                                      # self.net.pd: pred_shape})

        logging.info("Verification error= {:.1f}%, loss= {:.4f}, recall_c0= {:.4f}, recall_c1= {:.4f}".format(error_rate(prediction,
                                                                          util.crop_to_shape(batch_y,prediction.shape)),
                                                                          loss,
                                                                          recall_c0,
                                                                          recall_c1))
              
        img = util.combine_img_prediction(batch_x, batch_y, prediction)
        #util.save_image(img, "%s/%s.jpg"%(self.prediction_path, name))
        
        return pred_shape
    
    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.6f}".format(epoch, (total_loss / training_iters), lr))
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y,actual_loss):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions, recall_c0, recall_c1 = sess.run((self.summary_op, self.net.cost, self.net.accuracy, self.net.predicter, self.net.recall_c0, self.net.recall_c1),
                                                                             feed_dict={self.net.x: batch_x,
                                                                                        self.net.y: batch_y,
                                                                                        self.net.keep_prob: 1.,
                                                                                        self.net.mode: True,
                                                                                        self.learning_rate: self.lr})

        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Loss After Update= {:.4f},Recall_c0= {:.4f},Recall_c1= {:.4f},"
                     " Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                   actual_loss,
                                                                                   loss,
                                                                                   recall_c0,
                                                                                   recall_c1,
                                                                                   acc,
                                                                                   error_rate(predictions, batch_y)))

def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step+1)))) + (gradients[i] / (step+1))
        
    return avg_gradients

def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
