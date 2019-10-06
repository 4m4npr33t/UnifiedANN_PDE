import numpy as np

class sample_dataset:
    def __init__(self, dataset, batchsize, N):
        self. data = dataset
        self.batchsize = batchsize
        self.start_of_sample = 0
        self.end_of_sample = batchsize
        self.total_samples = int(len(self.data)/self.batchsize)
        self.current_sample = 0
        self.idx = N

    def get_new_sample(self):
        if self.end_of_sample > len(self.data):
            self.end_of_sample = len(self.data)-1

        ret = self.data[self.start_of_sample:self.end_of_sample]
        self.start_of_sample = self.end_of_sample
        self.end_of_sample += self.batchsize
        self.current_sample += 1
        if self.current_sample == self.total_samples:
            self.current_sample =0
            self.start_of_sample = 0
            self.end_of_sample = self.batchsize
            np.random.shuffle(self.data)
        return ret[:,0].reshape((ret.shape[0],1)), ret[:,self.idx].reshape((ret.shape[0],1))

