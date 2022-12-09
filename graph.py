import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["Step", "Value"]
df = pd.read_csv("okgood.csv", usecols=columns)
plt.plot(df.Step, df.Value)
plt.show()