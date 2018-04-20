from tf_unet import unet
import tensorflow as tf
images, mask = unet.get_image_mask()
image_batch_op, mask_batch_op = unet.batch_op(images, mask, batch_size=4)


with tf.Session() as sess:
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    init_op = tf.local_variables_initializer()
    sess.run(init_op)
    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # Get an image tensor and print its value.
    image, labels = sess.run([image_batch_op, mask_batch_op])
    print(labels.shape)
    coord.request_stop()
    print(labels.shape)