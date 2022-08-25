import torch
from utils import pickle_util, sample_util, worker_util
from torch.utils.data import DataLoader


def get_iter(batch_size, full_length, name2face_emb, name2voice_emb):
    train_iter = DataLoader(DataSet(name2face_emb, name2voice_emb, full_length),
                            batch_size=batch_size, shuffle=False, pin_memory=True, worker_init_fn=worker_util.worker_init_fn)
    return train_iter


class DataSet(torch.utils.data.Dataset):

    def __init__(self, name2face_emb, name2voice_emb, full_length):
        # 1. all movies
        self.train_movie_list = pickle_util.read_pickle("./dataset/voxceleb/cluster/train_movie_list.pkl")

        # 2.movie=>wav_list , movie=>jpg_list
        self.movie2wav_path = pickle_util.read_pickle("./dataset/voxceleb/cluster/movie2wav_path.pkl")
        self.movie2jpg_path = pickle_util.read_pickle("./dataset/voxceleb/cluster/movie2jpg_path.pkl")

        # 3.其它
        self.full_length = full_length
        self.name2face_emb = name2face_emb
        self.name2voice_emb = name2voice_emb

    def __len__(self):
        return self.full_length

    def __getitem__(self, index):
        # find a movie
        movie = sample_util.random_element(self.train_movie_list)
        # sample an image and a voice clip
        jpg = sample_util.random_element(self.movie2jpg_path[movie])
        wav = sample_util.random_element(self.movie2wav_path[movie])
        # to tensor
        jpg_emb = self.name2face_emb[jpg]
        wav_emb = self.name2voice_emb[wav]
        voice_tensor = torch.FloatTensor(wav_emb)
        face_tensor = torch.FloatTensor(jpg_emb)
        return voice_tensor, face_tensor
