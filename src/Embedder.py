import numpy as np


class Embedder:
    def __init__(self, id_map, embeddings):
        self.e = embeddings
        self.ind = id_map
        aid, iid = zip(*id_map.items())
        self.inv = dict(zip(iid, aid))

    def __getitem__(self, key):
        if type(key) == int:
            return self.e[self.ind[key], :]
        elif type(key) == np.ndarray:
            if isinstance(self.e, np.ndarray):
                return self.e[np.array([self.ind[k] for k in key], dtype=np.int32), :]
            else: # this is assumed to be tensorflow Variable, need to work out how to do type checking
                import tensorflow as tf
                slices = np.array([self.ind[k] for k in key], dtype=np.int32)
                return tf.gather(self.e, slices, axis=0)
        else:
            raise TypeError("Unknown type:", type(key))
