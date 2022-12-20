import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from datetime import datetime

# TODO redraw plots instead of creating new figures in loop


def getDescriptiveStatistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Min": round(dataframe.min(axis=0), 5),
        "Max": round(dataframe.max(axis=0), 5),
        "Mean": round(dataframe.mean(axis=0), 5),
        "Median": round(dataframe.median(axis=0), 5),
        "Mode": round(dataframe.mode(axis=0), 5).transpose().fillna("").astype(str).agg(";".join, axis=1),
        "1/4 Quantile": round(dataframe.quantile(0.25, axis=0), 5),
        "3/4 Quantile": round(dataframe.quantile(0.75, axis=0), 5),
        "Variance": round(dataframe.var(axis=0), 5),
        "Standard deviation": round(dataframe.std(axis=0), 5),
        "Range": round(dataframe.max(axis=0) - dataframe.min(axis=0), 5),
        "Coefficient of variation": round(dataframe.std(axis=0) / dataframe.mean(axis=0), 5),
        "Skewness": round(dataframe.skew(axis=0), 5),
        # "Kurtosis": round(dataframe.kurtosis(axis=0, bias=False), 5)
        "Kurtosis": np.round(stats.kurtosis(dataframe, axis=0, fisher=False), 5)
    })


def saveOrShowPlot(figure, savePath: str) -> None:
    if savePath:
        figure.set_size_inches(32, 18)
        figure.savefig(savePath + datetime.now().strftime("%Y%m%d-%H-%M-%S.%f") + '.png',
                       bbox_inches='tight')
    else:
        plt.show()


def createHistogramsAndBoxplots(dataframes: dict, savePath: str = None) -> None:
    for i in range(1, 6):
        fig = plt.figure()
        k = 1
        for key, df in dataframes.items():
            ax = fig.add_subplot(320 + k)
            ax.set_title(f"Histogram {key} dla kanału {i}")
            ax.hist(df[f"CH{i}"], bins=np.histogram_bin_edges(
                df[f"CH{i}"], bins='auto'))
            ax.grid(axis="y", linewidth=0.4)

            ax = fig.add_subplot(320 + k + 1)
            ax.set_title(f"Wykres pudełkowy {key} dla kanału {i}")
            ax.boxplot(df[f"CH{i}"])
            ax.grid(axis="y", linewidth=0.4)
            k += 2
        fig.subplots_adjust(left=0.125, bottom=0.08, right=0.9,
                            top=0.9, wspace=0.15, hspace=0.25)
        saveOrShowPlot(fig, savePath)


def _zeroCrossing(data: pd.DataFrame) -> dict:
    data = data.diff().dropna()
    zeros = {}
    for x in data:
        pos = data[x].values[data[x].values != 0.] > 0
        zeros[x] = len(np.where(np.bitwise_xor(pos[1:], pos[:-1]))[0])
    return zeros


def calculateAttributes(dataframe: pd.DataFrame) -> pd.DataFrame:
    sample = len(dataframe) // 300
    meanInWindowsDf = dataframe.groupby(dataframe.index // sample).mean()
    varInWindowsDf = dataframe.groupby(dataframe.index // sample).var().rename(columns={
        "CH1": "VarianceCH1", "CH2": "VarianceCH2", "CH3": "VarianceCH3", "CH4": "VarianceCH4", "CH5": "VarianceCH5"})

    meanCrossingCount = []
    zeroCrossingCount = []

    for name, group in dataframe.groupby(dataframe.index // sample):
        mask = ((group.shift() < meanInWindowsDf.iloc[name]) & (
                group > meanInWindowsDf.iloc[name]) | (group.shift() > meanInWindowsDf.iloc[name]) & (
                group < meanInWindowsDf.iloc[name]))
        meanCrossingCount.append(mask.sum().to_dict())
        zeroCrossingCount.append(_zeroCrossing(group))

    meanInWindowsDf = meanInWindowsDf.rename(
        columns={"CH1": "MeanCH1", "CH2": "MeanCH2", "CH3": "MeanCH3", "CH4": "MeanCH4", "CH5": "MeanCH5"})
    meanCrossingCountDf = pd.DataFrame(meanCrossingCount).rename(columns={
        "CH1": "CH1MCr", "CH2": "CH2MCr", "CH3": "CH3MCr", "CH4": "CH4MCr", "CH5": "CH5MCr"})
    zeroCrossingCountDf = pd.DataFrame(zeroCrossingCount).rename(columns={
        "CH1": "CH1ZCr", "CH2": "CH2ZCr", "CH3": "CH3ZCr", "CH4": "CH4ZCr", "CH5": "CH5ZCr"})
    seconds = pd.DataFrame({"Second": [i for i in range(1, 301)]})

    return pd.concat([seconds, meanInWindowsDf, varInWindowsDf,
                      meanCrossingCountDf, zeroCrossingCountDf], axis=1)


def createPlotsForOSWFiles(dataframes: dict, savePath: str = None) -> None:
    title = ["Wartości średniej w oknach 1 sekundowych dla kanału", "Wariancja w oknach 1 sekundowych dla kanału",
             "Liczba przejść przez wartość średnią w oknach 1 sekundowych dla kanału", "Liczba przejść przez zero pierwszej pochodnej w oknach 1 sekundowych dla kanału"]
    ylab = ["Średnia wartość", "Wariancja", "Liczba przejść przez wartość średnią",
            "Liczba przejść przez zero pierwszej pochodnej"]
    args = ["MeanCH", "VarianceCH", "CHMCr", "CHZCr"]
    keys = list(dataframes.keys())

    for i in range(4):
        for j in range(1, 6):
            fig, ax = plt.subplots()
            if args[i] == "CHMCr" or args[i] == "CHZCr":
                index = f"{args[i][0:2]}{j}{args[i][2:]}"
            else:
                index = f"{args[i]}{j}"
            p1, = ax.plot(
                dataframes[keys[0]].Second, dataframes[keys[0]][index], c="r", label=keys[0])
            p2, = ax.plot(
                dataframes[keys[1]].Second, dataframes[keys[1]][index], c="g", label=keys[1])
            p3, = ax.plot(
                dataframes[keys[2]].Second, dataframes[keys[2]][index], c="b", label=keys[2])

            plt.title(f"{title[i]} {j}")
            plt.xlabel("Sekunda")
            plt.ylabel(f"{ylab[i]}")
            plt.xlim(0, 300)
            plt.grid(linewidth=0.4)
            plt.legend(handles=[p1, p2, p3])

            saveOrShowPlot(fig, savePath)


def createPlotsForOSWChannels(dataframes: dict, savePath: str = None) -> None:
    title = ["Wartości średniej w oknach 1 sekundowych w pliku", "Wariancja w oknach 1 sekundowych w pliku",
             "Liczba przejść przez wartość średnią w oknach 1 sekundowych w pliku", "Liczba przejść przez zero pierwszej pochodnej w oknach 1 sekundowych w pliku"]
    ylab = ["Średnia wartość", "Wariancja", "Liczba przejść przez wartość średnią",
            "Liczba przejść przez zero pierwszej pochodnej"]
    args = ["MeanCH", "VarianceCH", "CHMCr", "CHZCr"]
    keys = list(dataframes.keys())
    for i in range(4):
        for j in range(3):
            fig, ax = plt.subplots()
            if args[i] == "CHMCr" or args[i] == "CHZCr":
                indexes = [
                    f"{args[i][0:2]}{k}{args[i][2:]}" for k in range(1, 6)]
            else:
                indexes = [f"{args[i]}{k}" for k in range(1, 6)]
            p1, = ax.plot(
                dataframes[keys[j]].Second, dataframes[keys[j]][indexes[0]], c="r", label="Kanał 1")
            p2, = ax.plot(
                dataframes[keys[j]].Second, dataframes[keys[j]][indexes[1]], c="g", label="Kanał 2")
            p3, = ax.plot(
                dataframes[keys[j]].Second, dataframes[keys[j]][indexes[2]], c="b", label="Kanał 3")
            p4, = ax.plot(
                dataframes[keys[j]].Second, dataframes[keys[j]][indexes[3]], c="orange", label="Kanał 4")
            p5, = ax.plot(
                dataframes[keys[j]].Second, dataframes[keys[j]][indexes[4]], c="darkorchid", label="Kanał 5")

            plt.title(f"{title[i]} {keys[j]}")
            plt.xlabel("Sekunda")
            plt.ylabel(f"{ylab[i]}")
            plt.xlim(0, 300)
            plt.grid(linewidth=0.4)
            plt.legend(handles=[p1, p2, p3, p4, p5])

            saveOrShowPlot(fig, savePath)


def _createLegendOnSubplot(figure, labels: list) -> None:
    ls = figure.add_subplot(326)
    ls.set_frame_on(False)
    ls.get_xaxis().set_visible(False)
    ls.get_yaxis().set_visible(False)
    ls.scatter([], [], marker=".", color="r", label=labels[0])
    ls.scatter([], [], marker=".", color="g", label=labels[1])
    ls.scatter([], [], marker=".", color="b", label=labels[2])
    ls.legend(loc="center")


def plotStatDispersionFigure(dataframes: dict, xArgs: list, yArgs: list, title: str, xlabel: str, ylabel: str, savePath: str = None) -> None:
    fig = plt.figure()
    fig.patch.set_edgecolor((0, 0, 0, 1.0))
    fig.patch.set_linewidth(1)
    keys = list(dataframes.keys())
    i = 1
    for x, y in zip(xArgs, yArgs):
        ax1 = fig.add_subplot(320 + i)
        ax1.set_title(f"{title} {i}")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.scatter(dataframes[keys[0]][x], dataframes[keys[0]][y],
                    c="r", marker=".",  label=keys[0])
        ax1.scatter(dataframes[keys[1]][x], dataframes[keys[1]][y],
                    c="g", marker=".", label=keys[1])
        ax1.scatter(dataframes[keys[2]][x], dataframes[keys[2]][y],
                    c="b", marker=".", label=keys[2])
        i += 1

    fig.subplots_adjust(left=0.1, bottom=0.06, right=0.93,
                        top=0.94, wspace=0.2, hspace=0.35)
    _createLegendOnSubplot(fig, keys)

    saveOrShowPlot(fig, savePath)


def createStatDispersionFigures(dataframes: dict, savePath: str = None) -> None:
    plotStatDispersionFigure(dataframes, [f"MeanCH{i}" for i in range(1, 6)], [f"VarianceCH{i}" for i in range(
        1, 6)], "Wykres średniej - wariancji w kanale", "Średnia", "Wariancja", savePath)

    plotStatDispersionFigure(dataframes, [f"MeanCH{i}" for i in range(1, 6)], [f"CH{i}MCr" for i in range(
        1, 6)], "Wykres średniej - liczby przejść przez średnią w kanale", "Średnia", "Liczba przejść przez średnią", savePath)

    plotStatDispersionFigure(dataframes, [f"MeanCH{i}" for i in range(1, 6)], [f"CH{i}ZCr" for i in range(
        1, 6)], "Wykres średniej - liczby przejść przez zero 1 pochodnej w kanale", "Średnia", "Liczba przejść przez zero", savePath)

    plotStatDispersionFigure(dataframes, [f"VarianceCH{i}" for i in range(1, 6)], [f"CH{i}MCr" for i in range(
        1, 6)], "Wykres wariancji - liczby przejść przez średnią w kanale", "Wariancja", "Liczba przejść przez średnią", savePath)

    plotStatDispersionFigure(dataframes, [f"VarianceCH{i}" for i in range(1, 6)], [f"CH{i}ZCr" for i in range(
        1, 6)], "Wykres wariancji - liczby przejść przez zero 1 pochodnej w kanale", "Wariancja", "Liczba przejść przez zero", savePath)

    plotStatDispersionFigure(dataframes, [f"CH{i}MCr" for i in range(1, 6)], [f"CH{i}ZCr" for i in range(
        1, 6)], "Wykres liczby przejść przez średnią - liczby przejść przez zero 1 pochodnej w kanale", "Liczba przejść przez średnią", "Liczba przejść przez zero", savePath)
