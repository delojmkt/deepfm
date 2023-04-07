from data.inputs import Input
import pandas as pd

__all__=["loading"]

class Loader(object):

    @staticmethod
    def _load(path, index):
        if index:
            return pd.read_csv(path)
        else:
            return pd.read_csv(path, index_col=[0])
    
    @staticmethod
    def load_df(path, index, target_nan):
        df = Loader._load(path, index)
        if target_nan:
            return Input().input_nan("inference",df)
        else:
            return Input().input_nan("training",df)

def loading(path, index=False, target_nan=False):
    return Loader.load_df(path, index, target_nan)