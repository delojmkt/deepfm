from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from data import Input

import pickle

import settings

__all__=['preprocess']

class Preprcoess(object):
    inputs = settings.input_col
    target = settings.target_col
    sparse_features = settings.sparse_features
    dense_features = settings.dense_features
    lbe = dict()

    def _preprocess(df, step):
        input_df = df[Preprcoess.inputs+Preprcoess.target].copy()
        # input_df = df[Preprcoess.inputs]
        # target_df = df[Preprcoess.target]

        if step == "_pre":
            for feat in Preprcoess.sparse_features:
                lbe = LabelEncoder()
                input_df[feat] = lbe.fit_transform(input_df[feat])
                Preprcoess.lbe.update({feat:lbe})

            with open("./lbe_dict.pickle","wb") as fw:
                pickle.dump(Preprcoess.lbe, fw)

        elif step == "_post":
            with open("./lbe_dict.pickle","rb") as f:
                Preprcoess.lbe_dict = pickle.load(f)
            for feat in Preprcoess.sparse_features:
                lbe = Preprcoess.lbe_dict[feat]
                input_df[feat] = lbe.transform(input_df[feat])
        
        mms = MinMaxScaler(feature_range=(0,1))
        # for feat in Preprcoess.dense_features:
        #     input_df[feat] = mms.fit_transform(input_df[feat])
        input_df[Preprcoess.dense_features] = mms.fit_transform(input_df[Preprcoess.dense_features])

        feature_names, feature_columns = Input().feature_names(input_df, Preprcoess.sparse_features, Preprcoess.dense_features)

        train, test = train_test_split(input_df, test_size=0.2)

        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        return feature_columns, (train_model_input, train[Preprcoess.target[0]].values), (test_model_input, test[Preprcoess.target[0]].values)
        
def preprocess(df, step_type):
    if step_type == "preprocess":
        return Preprcoess._preprocess(df,"_pre")
    elif step_type == "postprocess":
        return Preprcoess._preprocess(df,"_post")