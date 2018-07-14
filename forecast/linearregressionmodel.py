from sklearn.linear_model import LinearRegression

class LinearRegressionModel():

    def __init__(self):
        self.clf = LinearRegression(n_jobs=-1)

    def train(self, xTrain, yTrain, xTest, yTest):
        self.clf.fit(xTrain, yTrain)
        confidence = self.clf.score(xTest, yTest)

        return confidence

    def predict(self, x):
        prediction=self.clf.predict(x)

        return prediction