#!/usr/bin/env python3


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
import glob


class ColNames:
    """Dataframe column names"""

    INDEX = "index"
    PLAYER = "player"
    TEAM = "team"
    ROUND = "round"
    OPPENENT = "opponents"
    STAT = "stat"
    VALUE = "value"
    YEAR = "year"


def read_data(data_glob: str) -> pd.DataFrame:
    """Reads in list of parquet files and concatenates them into a single dataframe"""
    files = glob.glob(data_glob)
    years = [f.split("_")[-1].split(".")[0] for f in files]
    data = []

    for i, f in enumerate(files):
        df = pd.read_parquet(f, engine="fastparquet")
        df[ColNames.YEAR] = years[i]
        data.append(df)

    df = pd.concat(data, ignore_index=True)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe"""
    df = df.loc[(df[ColNames.VALUE] != "Off") & (df[ColNames.VALUE] != "On")]

    NaN_key = {"NA": np.NaN}
    df[ColNames.VALUE] = df[ColNames.VALUE].replace(NaN_key)
    df[ColNames.VALUE] = df[ColNames.VALUE].astype(np.float32)

    df = df.pivot(
        index=[
            ColNames.PLAYER,
            ColNames.TEAM,
            ColNames.ROUND,
            ColNames.OPPENENT,
            ColNames.YEAR,
        ],
        columns=ColNames.STAT,
        values=ColNames.VALUE,
    ).reset_index()

    df = df.drop(columns="subs").astype({"year": np.int32, "round": np.int32}).fillna(0)

    return df


def train_test_split(
    df: pd.DataFrame, data_start: int, test_year: int, last_round: int = 23
) -> (pd.DataFrame, pd.DataFrame):
    """Split the data into train and test sets"""
    df_test = df[(df["round"] < 23) & (df["year"] == test_year)]
    df_train = df[
        (df["round"] < 23) & (df["year"] >= data_start) & (df["year"] < test_year)
    ]

    # Get the labels for each stat
    xlabels = [
        "team",
        "opponents",
        "%_played",
        "behinds",
        "bounces",
        "clangers",
        "clearances",
        "contested_marks",
        "contested_possessions",
        "disposals",
        "frees",
        "frees_against",
        "goal_assists",
        "goals",
        "handballs",
        "hit_outs",
        "inside_50s",
        "kicks",
        "marks",
        "marks_inside_50",
        "one_percenters",
        "rebounds",
        "tackles",
        "uncontested_possessions",
    ]

    y_labels = "brownlow_votes"

    X_train = df_train.drop(
        columns=["player", "team", "opponents", "round", "year", "brownlow_votes"]
    )
    Y_train = df_train["brownlow_votes"]

    X_train, Y_train = torch.tensor(
        X_train.to_numpy(), dtype=torch.float
    ), torch.tensor(Y_train.to_numpy(), dtype=torch.float)

    return X_train, Y_train


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Sequential(nn.Linear(X.shape[1], 100), nn.ReLU())
        self.hidden2 = nn.Sequential(nn.Linear(100, 50), nn.ReLU())
        self.output = nn.Linear(50, 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x


def train(
    X,
    Y,
    loss_function=nn.MSELoss(),
    epoch_num=100,
    batch_size=100,
    lr=0.0001,
):
    loss = []
    network = Net()
    network.train()

    data_tuple = [[X[i], Y[i]] for i in range(len(X))]  # accuracy

    batch = torch.utils.data.DataLoader(data_tuple, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(network.parameters(), lr=lr, betas=(0.9, 0.999))

    for epoch in range(epoch_num):
        if not epoch % 10:
            print("Iteration: ", epoch, "Completion: ", (epoch) / epoch_num)

        running_loss = 0

        for batch_shuffle in batch:
            x, y = batch_shuffle
            y = y.unsqueeze(1)
            optimizer.zero_grad()
            loss = loss_function(network(x), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss.append(running_loss / batch_size)

    return network, loss


def main():
    data = clean_data(
        read_data("brownlow/data/afl_table_data/AFL-Player-Tables_game_stats*.parquet")
    )

    print(data.head())


if __name__ == "__main__":
    main()
