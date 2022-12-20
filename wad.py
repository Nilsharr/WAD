import pandas as pd
import scipy.io
import os
from wadFunctions import *
from pandas.plotting import autocorrelation_plot


directoryPath = os.getcwd()
chartSavePath = f"{directoryPath}\\charts\\temp\\"

mat1 = scipy.io.loadmat(f"{directoryPath}\\a01r.mat")
df1 = pd.DataFrame(mat1["a01r"], columns=["CH1", "CH2", "CH3", "CH4", "CH5"])

mat4 = scipy.io.loadmat(f"{directoryPath}\\a04r.mat")
df4 = pd.DataFrame(mat4["a04r"], columns=["CH1", "CH2", "CH3", "CH4", "CH5"])

mat7 = scipy.io.loadmat(f"{directoryPath}\\a07r.mat")
df7 = pd.DataFrame(mat7["a07r"], columns=["CH1", "CH2", "CH3", "CH4", "CH5"])

df1 = df1.dropna(how="any", axis=0)
df4 = df4.dropna(how="any", axis=0)
df7 = df7.dropna(how="any", axis=0)

# 2
# print(getDescriptiveStatistics(df1))
# print(getDescriptiveStatistics(df4))
# print(getDescriptiveStatistics(df7))

# createHistogramsAndBoxplots(
#     {"a01r": df1, "a04r": df4, "a07r": df7}, chartSavePath)

# 3
# print(df1.corr())
# print(df4.corr())
# print(df7.corr())

# 4 not working
# test = df1.groupby(
#     df1.index // 8192).sample(2048).reset_index(drop=True)

# autocorrelation_plot(test.CH1)
# autocorrelation_plot(df1.head(100000).CH2)
# plt.show()

# 5
# attDfs = {"a01r": calculateAttributes(df1), "a04r": calculateAttributes(
#     df4), "a07r": calculateAttributes(df7)}

# createPlotsForOSWFiles(attDfs, chartSavePath)
# createPlotsForOSWChannels(attDfs, chartSavePath)

# 6
# print('Scaled df by 2')
# scaledDf1By2 = df1.div(2)
# scaledDf4By2 = df4.div(2)
# scaledDf7By2 = df7.div(2)
# scaledDfsBy2 = {"a01r": calculateAttributes(scaledDf1By2), "a04r": calculateAttributes(
#     scaledDf4By2), "a07r": calculateAttributes(scaledDf7By2)}

# createPlotsForOSWFiles(scaledDfsBy2, chartSavePath)
# createPlotsForOSWChannels(scaledDfsBy2, chartSavePath)
# createStatDispersionFigures(scaledDfsBy2, chartSavePath)

# print('Scaled df by 16')
# scaledDf1By16 = df1.div(16)
# scaledDf4By16 = df4.div(16)
# scaledDf7By16 = df7.div(16)
# scaledDfsBy16 = {"a01r": calculateAttributes(scaledDf1By16), "a04r": calculateAttributes(
#     scaledDf4By16), "a07r": calculateAttributes(scaledDf7By16)}

# createPlotsForOSWFiles(scaledDfsBy16, chartSavePath)
# createPlotsForOSWChannels(scaledDfsBy16, chartSavePath)
# createStatDispersionFigures(scaledDfsBy16, chartSavePath)


# print('Downsampled df to 2048')
# downsampledDf1To2Ks = df1.groupby(
#     df1.index // 8192).sample(2048).reset_index(drop=True)
# downsampledDf4To2Ks = df4.groupby(
#     df4.index // 8192).sample(2048).reset_index(drop=True)
# downsampledDf7To2Ks = df7.groupby(
#     df7.index // 8192).sample(2048).reset_index(drop=True)
# downsampledDfsTo2Ks = {"a01r": calculateAttributes(downsampledDf1To2Ks), "a04r": calculateAttributes(
#     downsampledDf4To2Ks), "a07r": calculateAttributes(downsampledDf7To2Ks)}

# createPlotsForOSWFiles(downsampledDfsTo2Ks, chartSavePath)
# createPlotsForOSWChannels(downsampledDfsTo2Ks, chartSavePath)
# createStatDispersionFigures(downsampledDfsTo2Ks, chartSavePath)


# print('Downsampled df to 512')
# downsampledDf1ToHalfKs = df1.groupby(
#     df1.index // 8192).sample(512).reset_index(drop=True)
# downsampledDf4ToHalfKs = df4.groupby(
#     df4.index // 8192).sample(512).reset_index(drop=True)
# downsampledDf7ToHalfKs = df7.groupby(
#     df7.index // 8192).sample(512).reset_index(drop=True)
# downsampledDfsToHalfKs = {"a01r": calculateAttributes(downsampledDf1ToHalfKs), "a04r": calculateAttributes(
#     downsampledDf4ToHalfKs), "a07r": calculateAttributes(downsampledDf7ToHalfKs)}

# createPlotsForOSWFiles(downsampledDfsToHalfKs, chartSavePath)
# createPlotsForOSWChannels(downsampledDfsToHalfKs, chartSavePath)
# createStatDispersionFigures(downsampledDfsToHalfKs, chartSavePath)

# 8
# dfs = {"a01r": calculateAttributes(df1), "a04r": calculateAttributes(
#     df4), "a07r": calculateAttributes(df7)}

# createStatDispersionFigures(dfs, chartSavePath)
