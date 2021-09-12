from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np


def main(data, eta, epochs):
    df = pd.DataFrame(AND)
    print(df)
    X, y = prepare_data(df)
    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename="and.model")
    save_plot(df, "and.png", model)


if __name__ == 'main':  # << entry point
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1],
    }
    ETA = 0.3  # 0 and 1
    EPOCHS = 10

    main(AND, ETA, EPOCHS)