import gm_submodular
import gm_submodular.example_objectives as ex
from gm_submodular import leskovec_maximize

from video_sampling import *
import numpy as np
import scipy.spatial.distance as dist


class VSUM2(gm_submodular.DataElement):

    def __init__(self, video_id, seg_l=5):

        # load dataset data
        print(video_id)
        self.x, self.fnum, self.fps, self.length, self.img_id, self.idx = make_dsf(video_id)

        # budget 15% of orig
        self.budget = int(0.2 * self.length / seg_l)
        print ('budget: ', self.budget)

        # embed video segments
        # seg_feat = encodeSeg(self.dataset, model, seg_size=seg_l)

        # store segment features
        self.Y = np.ones(self.x.shape[0])

        # compute distance between segments
        self.dist_e = dist.squareform(dist.pdist(self.x, 'sqeuclidean'))

        # compute chronological distance
        # self.frame, self.score = self.dataset.sampleFrame()
        # self.frame = [self.img_id[i] for i in self.idx]
        # print(self.frame)

        fno_arr = np.expand_dims(np.array(self.img_id), axis=1)
        self.dist_c = dist.pdist(fno_arr, 'sqeuclidean')

    def getCosts(self):
        return np.ones(self.x.shape[0])

    # def getRelevance(self):
    #     return np.multiply(self.rel, self.rel)

    def getChrDistances(self):
        d = dist.squareform(self.dist_c)
        return np.multiply(d, d)

    def getDistances(self):
        return np.multiply(self.dist_e, self.dist_e)

    def summarizeRep(self, weights=[1.0, 0.0], seg_l=5):
        objectives = [representativeness(self),
                      uniformity(self)]

        selected, score, minoux_bound = leskovec_maximize(self,
                                                          weights,
                                                          objectives,
                                                          budget=self.budget)
        selected.sort()

        frames = []
        # gt_score = []
        for i in selected:
            frames.append(self.img_id[i:i + seg_l])
            # gt_score.append(self.score[i:i + seg_l])

        return selected, frames


def encodeSeg(data, model, seg_size=5):
    feat = data.feat

    img, img_id, score = data.sampleFrame()
    segs = [img_id[i:i + seg_size] for i in range(len(img_id) - seg_size + 1)]
    segs = reduce(lambda x, y: x + y, segs)

    x = feat[segs]

    # embedding
    enc_x = model(x)

    return enc_x.data


################################################
# objectives
################################################
def uniformity(S):
    '''
    Based on representativeness_shell implementation in 'example_objectives.py'
    :input S: DataElement with function getChrDistances()
    :return: uniformity objective
    '''
    tempDMat = S.getChrDistances()
    norm = tempDMat.mean()
    return (lambda X: (1 - ex.kmedoid_loss(X, tempDMat, float(norm))))


def representativeness(S):
    '''
    Based on representativeness_shell implementation in 'example_objectives.py'
    :input S: DataElement with function getDistances()
    :return: representativeness objective
    '''
    tempDMat = S.getDistances()
    norm = tempDMat.mean()
    return (lambda X: (1 - ex.kmedoid_loss(X, tempDMat, float(norm))))
