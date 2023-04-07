from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

import settings

__all__ = ["Input"]

class Input:
    def __init__(self):
        self.target = settings.target_col
        self.input = settings.input_col
        self.embed = settings.embedding_dim

    def input_nan(self, step, df):
        if step=="training":
            return df[~df[self.target[0]].isna()]
        elif step=="inference":
            return df[df[self.target[0]].isna()]

    def fixlen_feature_columns(self, df, sparse_features, dense_features):
        sparse_feat = [SparseFeat(feat, df[feat].max()+1, embedding_dim=self.embed) for feat in sparse_features]
        dense_feat = [DenseFeat(feat,1,) for feat in dense_features]
        return sparse_feat + dense_feat
    
    def feature_names(self, df, sparse_features, dense_features):
        fixlen_feature_columns = self.fixlen_feature_columns(df, sparse_features, dense_features)
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        return feature_names, (linear_feature_columns, dnn_feature_columns)
    



