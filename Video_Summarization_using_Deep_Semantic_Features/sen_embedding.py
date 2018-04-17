import numpy as np
import json
import skipthoughts
import random
from video_sampling import get_video_frames, get_one_video_frames
import nltk

# Set skip-thoughts model
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
nltk.download('punkt')

descfilename = 'data/videodatainfo_2017.json'


def get_random_id(num):
    videolist = []
    idx = 0
    while len(videolist) < 25:
        temp = random.randint(0, 9999)
        if temp != num:
            videolist.append('video'+str(temp))
            idx += 1
    return videolist


def make_pos_descriptions(video_id):
    json_data = open(descfilename).read()
    videodata = json.loads(json_data)

    for x in range(200000):
        if video_id == videodata["sentences"][x]["video_id"]:
            sentence = videodata["sentences"][x]["caption"]
            # print(video_id + ' ' + sentence)
            return sentence


def make_neg_descriptions(video_ids):
    json_data = open(descfilename).read()
    videodata = json.loads(json_data)
    neg_sentences = []
    for x in video_ids:
        for idx in range(200000):
            if x == videodata["sentences"][idx]["video_id"]:
                sentence = videodata["sentences"][idx]["caption"]
                # print(x + ' ' + sentence)
                neg_sentences.append(sentence)
                break

    return neg_sentences


def get_neg_sample(video_list):

    b_images = []
    b_desc = []
    b_t = []
    for video_id in video_list:
        images = get_video_frames(video_id)

        random_ids = get_random_id(str(video_id).replace("video", ""))
        neg_sentences = make_neg_descriptions(random_ids)

        descriptions = encoder.encode(neg_sentences)

        t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        b_images.append(images)
        b_desc.append(descriptions)
        b_t.append(t)

    v_images = np.concatenate(b_images)
    v_desc = np.concatenate(b_desc)
    v_t = np.concatenate(b_t)

    return v_images, v_desc, v_t


def get_pos_sample(video_list):

    b_images = []
    b_desc = []
    b_t = []
    for video_id in video_list:
        images = get_video_frames(video_id)

        pos_sentence = make_pos_descriptions(video_id)

        pos_sentences = [pos_sentence] * 25

        descriptions = encoder.encode(pos_sentences)

        t = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        b_images.append(images)
        b_desc.append(descriptions)
        b_t.append(t)

    v_images = np.concatenate(b_images)
    v_desc = np.concatenate(b_desc)
    v_t = np.concatenate(b_t)

    return v_images, v_desc, v_t


def make_pos_desc_images(video_list):

    images = []
    pos_sentences = []
    t = []

    for video_id in video_list:
        images.append(get_one_video_frames(video_id))
        pos_sentence = make_pos_descriptions(video_id)
        pos_sentences.append(pos_sentence)
        t.append(1)

    p_images = np.concatenate(images)

    descriptions = encoder.encode(pos_sentences)

    return p_images, descriptions, t


if __name__ == '__main__':

    video_path = 'data/test/anni001.mpg'
