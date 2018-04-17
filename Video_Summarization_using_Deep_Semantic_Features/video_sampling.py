import skvideo.io as sk
import numpy as np
import os
import json
import cv2
import random
import skipthoughts
import nltk
import tensorflow as tf
from functools import reduce


CHECKPOINT_PATH = 'tensorboard/checkpoints'
TRAIN_DATA_DIR = 'data/train-video'
TEST_DATA_DIR = 'data/test-video'
TRAIN_OUT_DIR = 'data/train_frame'
TEST_OUT_DIR = 'data/test_frame'
BATCH_SIZE = 64


def make_video_frames(datadir, outdir):
    # train video data
    for i in range(10000):
        if not os.path.exists(os.path.join(outdir, 'video%s' % i)):
            os.makedirs(os.path.join(outdir, 'video%s' % i))
        filename = ('video%s.mp4' % i)
        videopath = os.path.join(datadir, filename)
        cap = sk.vreader(videopath)

        metadata = sk.ffprobe(videopath)
        # print json.dumps(metadata["video"], indent=4)
        """
        fps : @r_frame_rate
        length : @duration
        frames : @nb_frames
        """
        length = float(json.dumps(metadata["video"]["@duration"]).replace('"', ''))
        frames = float(json.dumps(metadata["video"]["@nb_frames"]).replace('"', ''))
        fps = int(frames / length)

        print('%sth video' % i)
        print('length : %d / frames : %d / fps : %d' % (length, frames, fps))

        cent = np.linspace(0, length, 7)[1:-1]
        for x in range(len(cent)):
            cent[x] = int(cent[x])
        frames = cent * fps

        idx = 0
        filenum = 0
        for frame in cap:
            if idx in frames:
                frame = cv2.resize(frame, dsize=(224,224))
                sk.vwrite(outdir+'/video%s/frame%s.png' % (i, filenum), frame)
                filenum += 1
            idx += 1

        if i % 1000 == 0:
            print('%sth video processed...' % i)


def get_video_frames(video_id):
    dir = os.path.join(TRAIN_OUT_DIR,video_id)
    file_list = os.listdir(dir)

    arrays = []
    for file in file_list:
        fd = open(os.path.join(dir,file),'r')
        image = np.fromfile(fd,dtype=np.uint8)
        image = np.resize(image, [224, 224, 3]).astype(np.float32)
        arrays.append(image)

    images = np.stack(arrays)

    a_images = np.append(images, images, axis=0)

    for x in range(23):
        a_images = np.append(a_images, images, axis=0)

    return a_images


def get_one_video_frames(video_id):

    dir = os.path.join(TRAIN_OUT_DIR, video_id)
    file_list = os.listdir(dir)

    arrays = []
    for file in file_list:
        fd = open(os.path.join(dir,file),'r')
        image = np.fromfile(fd,dtype=np.uint8)
        image = np.resize(image, [224, 224, 3]).astype(np.float32)
        arrays.append(image)

    images = np.stack(arrays)

    return images


def make_dsf(video_path):

    save_path = os.path.join(CHECKPOINT_PATH, 'Video_Summarization_using_DSF-20001_step-20000.meta')
    model_path = os.path.join(CHECKPOINT_PATH, 'Video_Summarization_using_DSF-20001_step-20000')

    model = tf.train.import_meta_graph(save_path)

    frame_list, fnum, fps, length, img_id, idx = get_video_info(video_path)
    with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:

        model.restore(sess, model_path)

        dsf = []
        idxsize = float(np.shape(frame_list)[0])
        idx = np.round((idxsize/125)+0.5)
        idxsize = int(idxsize)
        idxlist = [x*125 for x in range(int(idx))]
        for index in idxlist:
            if index == idxlist[-1] and idxsize % 125 != 0:
                frames = frame_list[index:idxsize]
                zeros = np.zeros([index+125-idxsize, 224, 224, 3], dtype=np.float32)
                frames = np.concatenate((frames, zeros), axis=0)
            else:
                frames = frame_list[index:index+125]
            desc = np.zeros([25,4800], np.float32)
            t = np.zeros([25], np.float32)

            fc9 = sess.run(['modified_vgg/vgg_mod/X:0'], feed_dict={'Placeholders/inputimages:0': frames, 'Placeholders/descriptions:0': desc, 'Placeholders/negpos:0': t})
            dsf.append(fc9)

        dsf = np.reshape(dsf, [-1, 300])
    return dsf[:fnum], fnum, fps, length, img_id, idx


def get_video_info(video_path):

    cap = sk.vreader(video_path)
    seg_l = 4

    metadata = sk.ffprobe(video_path)
    # print (json.dumps(metadata, indent=4))
    # print (json.dumps(metadata["video"], indent=4))
    """
    fps : @r_frame_rate
    length : @duration
    frames : @nb_frames
    """
    length = float(json.dumps(metadata["video"]["@duration"]).replace('"', ''))
    # fnum = float(json.dumps(metadata["video"]["@nb_frames"]).replace('"', ''))
    fps = float(json.dumps(metadata["video"]["@r_frame_rate"]).replace('"', '').split('/')[0])/float(json.dumps(metadata["video"]["@r_frame_rate"]).replace('"', '').split('/')[1])
    fnum = int(np.ceil(length * fps))

    print('length : %.5f / frames : %d / fps : %.2f' % (length, fnum, fps))

    img_id = []
    frame_list = []
    id = 0
    for frame in cap:
        frame = cv2.resize(frame, dsize=(224, 224))
        frame_list.append(frame)
        img_id.append(id)
        id += 1

    segs = [img_id[i:i + seg_l] for i in range(len(img_id) - seg_l + 1)]
    segs = reduce(lambda x, y: x + y, segs)

    feat = []

    for seg in segs:
        feat.append(frame_list[seg])

    idx = np.arange(fps, fnum, fps)
    idx = np.floor(idx)
    idx = idx.tolist()
    idx = map(int, idx)

    return feat, fnum, fps, length, img_id, idx


if __name__ == '__main__':


    make_video_frames(TRAIN_DATA_DIR, TRAIN_OUT_DIR)

    # imagepath = 'data/testrv'
    #
    # folder_list = os.listdir(imagepath)
    #
    # for member in folder_list:
    #
    #     folder_path = os.path.join(imagepath, member)
    #     img_list = os.listdir(folder_path)
    #
    #     for file in img_list:
    #
    #         imgfile = os.path.join(folder_path, file)
    #         img = cv2.imread(imgfile)
    #         img = cv2.resize(img, dsize=(224, 224))
    #
    #         outpath = os.path.join(imagepath,file)
    #
    #         cv2.imwrite('%s' % outpath, img)



    # img = cv2.imread(os.path.join(imagepath, imagefile))
    # print(img)
    # print(np.shape(img))
    # img = cv2.resize(img, dsize=(224,224))

    # cv2.imwrite('data/resize_Irene.png', img)
    # video_path = 'data/test/anni001.mpg'
    #
    # dsf, fnum, fps, length, img_id = make_dsf(video_path)
    #
    # print(np.shape(dsf))
    # print(fnum)
    # print(fps)
    # print(length)
    # print(img_id)

    # dsf, fnum, fps = make_dsf('data/train-video/video3.mp4')
    #
    # print(dsf[0])

    # sentence = encoder.encode(['a girl is singing on the stage.'])
    # print(sentence)

    # imgs, desc, t = get_images_descriptions(['video0', 'video1'])
    #
    # print(np.shape(imgs))
    # print(np.shape(desc))
    # print(t)

    # n_imgs, n_desc, n_t = get_neg_sample(['video1'])
    #
    # p_imgs, p_desc, p_t = get_pos_sample(['video1'])
    #
    # print(np.shape(n_imgs))
    # print(np.shape(p_imgs))
    # print(n_imgs[0])
    # print(p_imgs[0])
    # print(np.shape(n_desc))
    # print(np.shape(p_desc))
    # print(n_desc[:2])
    # print(p_desc[:2])
    # print(n_t)

    # fps = 29.97
    # fnum = 2356
    #
    # idx = np.arange(fps, fnum, fps)
    # idx = np.floor(idx)
    # idx = idx.tolist()
    # idx = map(int, idx)
    # print(idx)

    # frame_list, fnum, fps, length, img_id, idx = make_dsf('data/test-video/video10093.mp4')