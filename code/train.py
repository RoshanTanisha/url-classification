from load_data import load_files
from models import get_SVM, get_rnn_layers
from sklearn.metrics import accuracy_score, precision_score


def train(model, train_x, train_y, reshape=False):
    if reshape:
        return model.fit(train_x.reshape((-1, 1, 50)), train_y)
    return model.fit(train_x, train_y)


def predict(model, test_x, reshape=False):
    if reshape:
        return model.predict(test_x.reshape((-1, 1, 50)))
    return model.predict(test_x)


def evaluate(test_y, pred_y, reshape=False):
    if reshape:
        pred_y = list(map(lambda x: [int(x[0] > 0.5)], pred_y))
    print("Accuracy: ", accuracy_score(test_y, pred_y))
    print('Precision: ', precision_score(test_y, pred_y))


if __name__ == "__main__":
    trainx, trainy, testx, testy = load_files(1, 50, False)
    # model = get_rnn_layers(50)
    model = get_SVM()
    # print(trainx.reshape((-1, 1, 50)).shape)
    # print(testx.reshape((-1, 1, 50)).shape)
    train(model, trainx, trainy, reshape=False)
    evaluate(testy, predict(model, testx, reshape=False), reshape=False)
