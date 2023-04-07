from deepctr.models import DeepFM
from sklearn.metrics import r2_score

class DeepFMModel:
    def __init__(self,feature_columns,task='regression'):
        self.feature_columns = feature_columns
        self.task = task
        self.optimizer = optimizer
        self.loss = loss
        self.metrics_list = metrics_list
        self.epochs = epochs
        self.model = None


    def _build(self,feature_columns, task, optimizer='adam', loss='mae', metrics_list=['mae']):
        model = DeepFM(*feature_columns, task=task)
        model.compile(optimizer, loss, metrics=[metrics_list])

        return model
    
    def train(self,input,target,epochs):
        self.model = self._build(self.feature_columns, self.task)
        hist = self.model.fit(input,target,epochs)

        self._save(self.model, )

        print("============================================================")
        

        return hist

    def predict(self,input_df):
        return self.model.predict(input_df)

    def evaluate(train_input,train_target):
        predict = self.predict(train_input)
        print("R2 Score: ", round(r2_score(predict, train_target),4))

    def _save(self, path):
        self.model.save(path)