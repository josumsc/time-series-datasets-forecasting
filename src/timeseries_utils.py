import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def timeseries_csv_to_df(
    file: str,
    timeseries_index: int=0,
    timeseries_format: str=None
) -> pd.DataFrame:
    """Reads a CSV file as a dataset while converting the timeseries col to the correct type

    :param file: Path to the file to read
    :param timeseries_index: Index of the column containing the time data
    :param timeseries_format: Format of the datetime column
    :return: Pandas DataFrame
    """
    df = pd.read_csv(file)
    if timeseries_format:
        df.iloc[:, timeseries_index] = pd.to_datetime(df.iloc[:, timeseries_index], format=timeseries_format)
    else:
        df.iloc[:, timeseries_index] = pd.to_datetime(df.iloc[:, timeseries_index])
        
    df.set_index(df.columns[timeseries_index], inplace=True)

    return df


def split_train_test(
    dataset: pd.Series,
    train_size: float=0.8
) -> tuple:
    """Splits the dataset into train and test subsets
    
    :param dataset: Pandas Series with the information
    :param train_size: Percentage of data used in training
    :return: Tuple with the train and test datasets
    """
    train_samples = int(np.floor(len(dataset)*train_size))
    train = dataset[:train_samples]
    test = dataset[train_samples:]
    
    return train, test


def plot_results(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prediction: pd.DataFrame,
    figsize: tuple=(16, 8)
) -> None:
    """Plots the results of our forecasting
    :param train: Train data
    :param test: Test data
    :param prediction: Predictions done
    :return: None
    """
    plt.figure(figsize=figsize)
    plt.plot(train,label="Training")
    plt.plot(test,label="Test")
    plt.plot(prediction,label="Predicted")
    plt.legend(loc = 'upper left')
    plt.show()
    