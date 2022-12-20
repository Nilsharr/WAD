import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import autocorrelation_plot

directoryPath = os.getcwd()

dataframe = pd.read_csv(f"{directoryPath}\\msc.csv", sep=";", index_col=False)

dataframe = dataframe.dropna(how='any', axis=0)

# 2

so = pd.DataFrame({
    'Min': round(dataframe.Snvalue.min(), 3),
    'Max': round(dataframe.Snvalue.max(), 3),
    'Mean': round(dataframe.Snvalue.mean(), 3),
    'Median': round(dataframe.Snvalue.median(), 3),
    'Mode': round(dataframe.Snvalue.mode(), 3).to_frame().T.astype(str).agg(';'.join, axis=1),
    '1/4 Quantile': round(dataframe.Snvalue.quantile(0.25), 3),
    '3/4 Quantile': round(dataframe.Snvalue.quantile(0.75), 3),
    'Variance': round(dataframe.Snvalue.var(), 3),
    'Standard deviation': round(dataframe.Snvalue.std(), 3),
    'Skewness': round(dataframe.Snvalue.skew(), 3),
    'Kurtosis': round(dataframe.Snvalue.kurtosis(), 3)
})
print(so)

# plt.plot(dataframe.year, dataframe.Snvalue)
# plt.title("Średnia miesięczna całkowitej liczby plam słonecznych")
# plt.xlim(dataframe.year.values[0], dataframe.year.values[-1])
# plt.ylim(0)
# plt.grid(axis="y", linestyle="-", linewidth=0.4)
# plt.ylabel("Liczba plam słonecznych Sn")
# plt.xlabel("Czas [lata]")
# plt.show()

# 3

# span = int(np.ceil((dataframe.year.values[-1] - dataframe.year.values[0]) / 5))
# means = [dataframe.Snvalue.mean()]
# stds = [dataframe.Snvalue.std()]
# labels = [f"{dataframe.year.values[0]}-{dataframe.year.values[-1]}"]
# for i in range(dataframe.year.values[0], dataframe.year.values[-1], span):
#     data = dataframe.loc[(dataframe.year >= i) & (dataframe.year < i + span)]
#     means.append(data.Snvalue.mean())
#     stds.append(data.Snvalue.std())
#     labels.append(f"{data.year.values[0]}-{data.year.values[-1]}")
#     print(pd.DataFrame({
#         'Span': f'{data.year.values[0]} - {data.year.values[-1]}',
#         'Min': round(data.Snvalue.min(), 3),
#         'Max': round(data.Snvalue.max(), 3),
#         'Mean': round(data.Snvalue.mean(), 3),
#         'Median': round(data.Snvalue.median(), 3),
#         'Mode': round(data.Snvalue.mode(), 3).to_frame().T.astype(str).agg(';'.join, axis=1),
#         '1/4 Quantile': round(data.Snvalue.quantile(0.25), 3),
#         '3/4 Quantile': round(data.Snvalue.quantile(0.75), 3),
#         'Variance': round(data.Snvalue.var(), 3),
#         'Standard deviation': round(data.Snvalue.std(), 3),
#         'Skewness': round(data.Snvalue.skew(), 3),
#         'Kurtosis': round(data.Snvalue.kurtosis(), 3)
#     }))

# plt.bar(labels, means, color="navy", width=0.4)
# plt.title("Średnia liczby plam słonecznych w okresach")
# plt.grid(axis="y", linestyle="-", linewidth=0.4)
# plt.ylabel("Średnia liczby plam słonecznych")
# plt.xlabel("Okres")
# for i in range(len(means)):
#     plt.text(i, means[i], round(means[i], 2), ha='center')
# plt.show()

# plt.bar(labels, stds, color="indigo", width=0.4)
# plt.title("Odchylenie standardowe liczby plam słonecznych w okresach")
# plt.grid(axis="y", linestyle="-", linewidth=0.4)
# plt.ylabel("Odchylenie standardowe liczby plam słonecznych")
# plt.xlabel("Okres")
# for i in range(len(stds)):
#     plt.text(i, stds[i], round(stds[i], 2), ha='center')
# plt.show()

# 4

# centuries = [dataframe.loc[(dataframe.year < 1801)].Snvalue]
# for i in range(1801, dataframe.year.values[-1], 100):
#     centuries.append(dataframe.loc[(dataframe.year >= i)
#                                    & (dataframe.year < i + 100)].Snvalue)
# plt.boxplot(centuries, labels=["XVIII w.", "XIX w.", "XX w.", "XXI w."])
# plt.title("Porównanie plam na słońcu przez stulecia")
# plt.ylabel("Liczba plam słonecznych Sn")
# plt.xlabel("Wiek")
# plt.grid(axis="y", linewidth=0.4)
# plt.show()

# 5
# Freedman-Diaconis rule

# bin_width = 2 * (dataframe.Snvalue.quantile(0.75) -
#                  dataframe.Snvalue.quantile(0.25)) / (len(dataframe.Snvalue) ** (1 / 3))
# bin_count = int(
#     np.ceil((dataframe.Snvalue.max() - dataframe.Snvalue.min()) / bin_width))
# plt.hist(dataframe.Snvalue, bins=bin_count, edgecolor="black")
# plt.title("Histogram plam słonecznych")
# plt.xlim(0)
# plt.ylabel("Ilość plam słonecznych w przedziale")
# plt.xlabel(
#     f"Przedziały wartości liczb plam słonecznych Sn (h = {int(bin_width)})")
# plt.show()

# 6
# https://www.alpharithms.com/autocorrelation-time-series-python-432909/

# autocorrelation_plot(dataframe.Snvalue)
# plt.xticks([i for i in range(0, 3500, 250)])
# plt.grid(axis="x", linewidth=0.4)
# plt.show()
