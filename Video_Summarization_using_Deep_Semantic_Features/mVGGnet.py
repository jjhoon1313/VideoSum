import vgg16
import time
from tensorflow.python.client import device_lib

from video_sampling import *
from sen_embedding import get_pos_sample, get_neg_sample


def fc(inp, layers, activation=None, name=None):
    net = tf.layers.dense(inputs=inp,
                          units=layers,
                          activation=activation,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          bias_initializer=tf.zeros_initializer(),
                          name=name)

    return net

def get_available_gpus():
    local_devices = device_lib.list_local_devices()
    return [x.name for x in local_devices if x.device_type == 'GPU']

def model(img, desc, reuse=False):

    # Set VGG / Modified VGG Graph
    vgg = vgg16.Vgg16()
    with tf.name_scope('content_vgg'):
        vgg.build(img)

        fc7 = vgg.relu7

    with tf.variable_scope('modified_vgg'):
        with tf.name_scope('vgg_mod'):
            fc8 = fc(fc7, 1000, activation=tf.nn.tanh, name='mfc8')
            fc9 = fc(fc8, 300, activation=tf.nn.tanh, name='mfc9')
            last = tf.reshape(fc9, [-1, 5, 300])
            x = tf.reduce_mean(last, axis=1, name='X')

        with tf.name_scope('description_network'):
            fc2_1 = fc(desc, 1000, activation=tf.nn.tanh, name='mfc2_1')
            y = fc(fc2_1, 300, activation=tf.nn.tanh, name='mfc2_2')

    return x, y

if __name__ == '__main__':

    gpu_names = get_available_gpus()
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected.'.format(gpu_num))

    device1 = gpu_names[0]
    device2 = gpu_names[1] if int(gpu_num) == 2 else device1

    print('device1: {0}'.format(device1))
    print('device2: {0}'.format(device2))

    # Path for tf.summary.FileWriter and to store model checkpoints
    FILEWRITER_PATH = 'tensorboard'
    CHECKPOINT_PATH = 'tensorboard/checkpoints'

    BATCH_SIZE = 25
    BATCH_COUNT = BATCH_SIZE / 25
    EPOCH_NUM = 2

    # tf.graph, placeholders
    tf.reset_default_graph()

    with tf.name_scope('Placeholders'):
        images_ = tf.placeholder("float", [None, 224, 224, 3], name='inputimages')
        desc = tf.placeholder(tf.float32, [None, 4800], name='descriptions')
        t = tf.placeholder(tf.float32, [None], name='negpos')

        images_A = tf.split(images_, int(gpu_num))
        desc_A = tf.split(desc, int(gpu_num))
        t_A = tf.split(t, int(gpu_num))



    # Set Loss
    for gpu_id in range(int(gpu_num)):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                X, Y = model(images_A[gpu_id], desc_A[gpu_id], gpu_id > 0)
                dXY = tf.reduce_sum(tf.square(tf.subtract(X, Y)), axis=1)
                alpha = tf.constant(17.52208138, dtype=tf.float32, name='alpha')
                loss = tf.reduce_sum(tf.add(tf.multiply(0.1*t_A[gpu_id], dXY),
                                            tf.multiply((1-t_A[gpu_id])*0.9,
                                                        tf.maximum(0.0, tf.subtract(alpha, dXY)))))

    t_vars = tf.trainable_variables()

    v_vars = [var for var in t_vars if 'modified_vgg' in var.name]

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=v_vars)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Training Start
    # Recover all weight variables from the last checkpoint
    RECOVER_CKPT = False

    # tf.summary
    train_writer = tf.summary.FileWriter('logdir/train', graph=tf.get_default_graph())

    with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.8)),
                                          log_device_placement=True)) as sess:

        # (optional) load model weights
        if RECOVER_CKPT:
            latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_PATH)
            print("Loading the last checkpoint: " + latest_ckpt)
            saver.restore(sess, latest_ckpt)
            last_epoch = int(latest_ckpt.replace('_', '*').replace('-', '*').split('*')[4])
        else:
            last_epoch = 0

        sess.run(tf.global_variables_initializer())
        step = 0

        for epoch in range(last_epoch, EPOCH_NUM):

            print('Training Start... Epoch : %s' % epoch)

            total_loss = 0
            train_elapsed = []

            for f in range(10000 / BATCH_COUNT):

                t_start = time.time()
                v_list = []
                for i in range(BATCH_COUNT):
                    v_list.append('video'+str(BATCH_COUNT*f+i))

                p_imgs, p_desc, p_t = get_pos_sample(v_list)
                n_imgs, n_desc, n_t = get_neg_sample(v_list)

                _, p_loss_, p_dsf, p_y_, p_dXY_ = sess.run([optimizer, loss, X, Y, dXY],
                                                   feed_dict={images_: p_imgs, desc: p_desc, t: p_t})

                _, n_loss_, n_dsf, n_y_, n_dXY_ = sess.run([optimizer, loss, X, Y, dXY],
                                                   feed_dict={images_: n_imgs, desc: n_desc, t: n_t})

                total_loss += p_loss_ + n_loss_
                t_elapsed = time.time() - t_start
                train_elapsed.append(t_elapsed)

                step += 1

                if step % 100 == 0:
                    print(('[trn] step {:d}, loss {:f}, sec/iter {:f}').format(step,
                                                                               total_loss/(step-(epoch-last_epoch)*10000),
                                                                               np.sum(train_elapsed)))

                if train_writer:
                    summary = tf.Summary(value=[tf.Summary.Value(tag='P_LOSS', simple_value=p_loss_),
                                                tf.Summary.Value(tag='N_LOSS', simple_value=n_loss_)])
                    train_writer.add_summary(summary, global_step=step)

            # save checkpoint of the model at each 5000 steps
                if step % 5000 == 0:
                    print("Saving checkpoint of model...")
                    checkpoint_name = os.path.join(CHECKPOINT_PATH, 'Video_Summarization_using_DSF2-' + str(step + 1) + '_step')
                    save_path = saver.save(sess, checkpoint_name, global_step=step)
                    print("Step: %d, Model checkpoint saved at %s" % (step + 1, checkpoint_name + '-(#global_step)'))