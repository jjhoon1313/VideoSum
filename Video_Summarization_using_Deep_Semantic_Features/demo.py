from func.nets import vid_enc, vid_enc_vgg19
from chainer import serializers, configuration
from summarize import get_flabel
from vsum2 import VSUM2
from skvideo.io import vread, vwrite

import chainer
import numpy as np
import os
import json
import sys
sys.path.append('script/')


# prepair output dir
d_name = 'vset'
out_dir = 'results/{:}/'.format(d_name)
print ('save to: ', out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# summarize video
v_id = "video0"
# video_path = 'data/summe/videos/%s.mp4' % v_id
video_path = 'data/%s.mp4' % v_id

with configuration.using_config('train', False):
    with chainer.no_backprop_mode():
        vsum = VSUM2(video_path, seg_l=5)

selected, frames = vsum.summarizeRep(seg_l=5, weights=[1.0, 0.0])

# get 0/1 label for each frame
fps = vsum.fps
fnum = vsum.fnum

print('selected : %s' % selected)
print('frames : %s ' % frames)

label = get_flabel(frames, fnum, fps, seg_l=5)

# write summarized video

video_data = vread(video_path)

sum_vid = video_data[label.ravel().astype(np.bool), :,:,:]

print('writing video to', 'sum_%s.mp4'%v_id)
vwrite('sum_%s.mp4'%v_id, sum_vid)