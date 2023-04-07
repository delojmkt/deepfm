import pandas as pd
from data import loading, Input
from data import preprocess
import settings

if __name__=="__main__":

    path = "/Users/gimda-eun/Downloads/laptops.csv"

    print("data load")
    df = loading(path)    # train data
    # df = loading(path,target_nan=True) ## inference data
    print(df.head())

    print("preprocess")
    feature_columns, train, test = preprocess(df, "preprocess")
    print("feature_columns: ",feature_columns)
    print("train:" , train)
    print("test: ", test)
