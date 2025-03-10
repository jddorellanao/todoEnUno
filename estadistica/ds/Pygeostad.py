# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 22:06:40 2024

@author: Daniel Vazquez-Ramirez

This code was developed using Python v3.9
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.stats as stats
import os
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from statsmodels.multivariate.manova import MANOVA
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from collections import Counter
#from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
from scipy.stats import kstest
from scipy.stats import anderson
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2, spearmanr, kendalltau, pearsonr
from sklearn.exceptions import ConvergenceWarning
from scipy.spatial.distance import mahalanobis


os.environ["OMP_NUM_THREADS"] = "2"
##########################################################################################################

def Estadisticas(data, variable, save_path, filename, lan='esp'):
    """
    Computes descriptive statistics for a given variable in the dataset and saves the results to a CSV file.

    Parameters:
    - data: pd.DataFrame, the dataset containing the variable.
    - variable: str, the column name for the variable to analyze.
    - save_path: str, the directory where the CSV file will be saved.
    - filename: str, the name of the output CSV file.
    - lan: str, the language for the output ('esp' for Spanish, 'eng' for English).

    Returns:
    - pd.Series containing descriptive statistics for the variable.
    """
    # Extract the column
    if variable not in data.columns:
        raise ValueError(f"Column '{variable}' not found in the dataset.")
    
    x = data[variable]
    
    if not pd.api.types.is_numeric_dtype(x):
        raise ValueError(f"Column '{variable}' must be numeric.")
    
    if x.empty:
        raise ValueError(f"Column '{variable}' is empty.")
    
    # Calculate statistics
    Stat1 = x.describe()
    ran = max(x) - min(x)  # Range
    IQR = Stat1['75%'] - Stat1['25%']  # Interquartile range
    var = x.var()  # Variance
    Skew = x.skew()  # Skewness
    curtosis = x.kurt()  # Kurtosis

    # Create the statistics list
    Stat = [
        x.name, Stat1['count'], Stat1['min'], Stat1['25%'], Stat1['50%'],
        Stat1['mean'], Stat1['75%'], Stat1['max'], ran, IQR, var,
        Stat1['std'], Skew, curtosis
    ]

    # Convert to Series
    Stat = pd.Series(Stat)
    
    # Rename indices based on the language
    if lan == 'esp':
        Stat.rename(index={
            0: 'Estadisticas', 1: 'Muestras', 2: 'Mínimo', 3: '1er cuartil',
            4: 'Mediana', 5: 'Media', 6: '3er cuartil', 7: 'Máximo',
            8: 'Rango', 9: 'Rango intercuartil', 10: 'Varianza',
            11: 'Desviación estándar', 12: 'Simetria', 13: 'Curtosis'
        }, inplace=True)
    elif lan == 'eng':
        Stat.rename(index={
            0: 'Statistics', 1: 'Samples', 2: 'Minimum', 3: '1st quartile',
            4: 'Median', 5: 'Mean', 6: '3rd quartile', 7: 'Maximum',
            8: 'Range', 9: 'IQR', 10: 'Variance', 11: 'STD',
            12: 'Skewness', 13: 'Kurtosis'
        }, inplace=True)
    else:
        raise ValueError("Invalid language. Use 'esp' for Spanish or 'eng' for English.")
    
    # Save the statistics to CSV
    file_path = f"{save_path}/{filename}.csv"
    Stat.to_csv(file_path, header=True)
    
    return Stat

##########################################################################################################

def EstadisticasDF(df, save_path, filename, lan='eng', variables=None):
    """
    Calculate descriptive statistics for selected numerical columns in a DataFrame and save the results to a CSV file.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing numerical data.
    save_path (str): The directory where the CSV file will be saved.
    filename (str): The name of the output CSV file.
    lan (str, optional): The language for the output statistics. 
                         'eng' for English (default) or 'esp' for Spanish.
    variables (list, optional): A list of column names to include in the analysis.
                                If None, all numerical columns are included.

    Returns:
    pd.DataFrame: A DataFrame containing descriptive statistics for each numerical column.
    """
    # Ensure variables is a list if provided, otherwise include all numerical columns
    if variables is None:
        variables = df.select_dtypes(include=['number']).columns.tolist()
    
    # Initialize an empty dictionary to store statistics for each selected column
    stats_dict = {}
    
    for col in variables:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
        
        x = df[col]
        Stat1 = x.describe()

        ran = max(x) - min(x)  # Range
        IQR = Stat1['75%'] - Stat1['25%']  # Interquartile range
        var = x.var()  # Variance
        Skew = x.skew()  # Skewness
        curtosis = x.kurt()  # Kurtosis

        stats_dict[col] = [
            Stat1['count'], Stat1['min'], Stat1['25%'], Stat1['50%'],
            Stat1['mean'], Stat1['75%'], Stat1['max'], ran, IQR, var,
            Stat1['std'], Skew, curtosis
        ]
    
    # Transpose the dictionary to create a DataFrame with rows as statistics
    Stat_df = pd.DataFrame.from_dict(stats_dict, orient='index').transpose()

    if lan == 'esp':
        Stat_df.index = ['Muestras', 'Mínimo', '1er cuartil', 'Mediana', 'Media',
                         '3er cuartil', 'Máximo', 'Rango', 'Rango intercuartil', 'Varianza',
                         'Desviación estándar', 'Simetría', 'Curtosis']
    else:
        Stat_df.index = ['Samples', 'Minimum', '1st quartile', 'Median', 'Mean',
                         '3rd quartile', 'Maximum', 'Range', 'IQR', 'Variance',
                         'STD', 'Skewness', 'Kurtosis']

    # Save the statistics to CSV
    file_path = f"{save_path}/{filename}.csv"
    Stat_df.to_csv(file_path)

    return Stat_df
##########################################################################################################

def Estadisticas_grouped(data, variable, group, save_path, filename, lan='eng'):
    """
    Calculate descriptive statistics for a specified variable in a DataFrame, including grouped statistics,
    and save the results to a CSV file.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data to be analyzed.
    variable (str): The name of the numeric column for which statistics will be calculated.
    group (str): The name of the column by which the data will be grouped before computing statistics.
    save_path (str): The directory where the CSV file will be saved.
    filename (str): The name of the output CSV file.
    lan (str, optional): The language for the output statistics. 
                        'esp' for Spanish (e.g., "Media", "Desviación estándar"),
                        'eng' for English (default, e.g., "Mean", "Standard Deviation").

    Returns:
    pd.DataFrame: A DataFrame containing descriptive statistics for the specified variable,
                  including statistics for each group defined by the `group` column.
    """
    x = data[variable]
    Stat1 = x.describe()

    ran = max(x) - min(x)  # Range
    IQR = Stat1['75%'] - Stat1['25%']
    var = x.var()  # Variance
    Skew = x.skew()  # Skewness
    curtosis = x.kurt()

    Stat = [
        x.name, Stat1['count'], Stat1['min'], Stat1['25%'], Stat1['50%'],
        Stat1['mean'], Stat1['75%'], Stat1['max'], ran, IQR, var, Stat1['std'], Skew, curtosis
    ]

    # Create a DataFrame for the statistics
    if lan == 'esp':
        index = ['Estadísticas', 'Muestras', 'Mínimo', '1er cuartil', 'Mediana', 'Media', '3er cuartil',
                 'Máximo', 'Rango', 'Rango intercuartil', 'Varianza', 'Desviación estándar', 'Simetría', 'Curtosis']
    elif lan == 'eng':
        index = ['Statistics', 'Samples', 'Minimum', '1st quartile', 'Median', 'Mean', '3rd quartile',
                 'Maximum', 'Range', 'IQR', 'Variance', 'STD', 'Skewness', 'Kurtosis']
    else:
        raise ValueError("Invalid language. Use 'esp' for Spanish or 'eng' for English.")

    result = pd.DataFrame({"Variable": pd.Series(Stat, index=index)})

    # Compute statistics for each group
    grouped = data.groupby(group)

    for group_name, group_data in grouped:
        x = group_data[variable]
        Stat1 = x.describe()

        ran = max(x) - min(x) if not x.empty else 0  # Range
        IQR = Stat1['75%'] - Stat1['25%'] if not x.empty else 0
        var = x.var() if len(x) > 1 else 0  # Variance
        Skew = x.skew() if not x.empty else 0  # Skewness
        curtosis = x.kurt() if not x.empty else 0  # Kurtosis

        Stat = [
            x.name, Stat1['count'], Stat1['min'], Stat1['25%'], Stat1['50%'],
            Stat1['mean'], Stat1['75%'], Stat1['max'], ran, IQR, var, Stat1['std'], Skew, curtosis
        ]

        result[group_name] = pd.Series(Stat, index=index)

    # Save the statistics to CSV
    file_path = f"{save_path}/{filename}.csv"
    result.to_csv(file_path)

    return result

##########################################################################################################

def HistBoxplot(data, variable, bins, stat, color, xlab, title, lan, save_path=None, filename=None):
    """
    Create a combined histogram and boxplot with calculated mean and median, and save the plot as a PNG file.

    Parameters:
    - data (DataFrame): Dataset containing the variable to plot.
    - variable (str): Column name of the variable to analyze.
    - bins (int): Number of bins for the histogram.
    - stat (str): Type of histogram statistics (e.g., 'count', 'density').
    - color (str): Color for the plots.
    - xlab (str): Label for the x-axis.
    - title (str): Title for the plot.
    - lan (str): Language for labels ('esp' for Spanish, 'eng' for English).
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """
    x = data[variable].dropna()
    mean = np.mean(x)  # Calculate mean
    median = np.median(x)  # Calculate median

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                        gridspec_kw={"height_ratios": (.15, .85)})
    
    ax_box.set_title(title)
    sns.boxplot(x=x, orient='h', color=color, notch=True, ax=ax_box)
    ax_box.axvline(mean, ymin=0, ymax=1, color='red', linestyle="dashed")
    ax_box.axvline(median, ymin=0, ymax=1, color='blue', linestyle="dashed")

    sns.set_style("whitegrid", {'grid.linestyle': '-.'})
    sns.histplot(x, bins=bins, stat=stat, color=color, edgecolor='black', ax=ax_hist)
    ax_hist.axvline(mean, ymin=0, ymax=1, color='red', linestyle="dashed")
    ax_hist.axvline(median, ymin=0, ymax=1, color='blue', linestyle="dashed")

    for ax in [ax_box, ax_hist]:
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             color='black', linewidth=2, fill=False)
        ax.add_patch(rect)

    plt.xlabel(xlab)
    if lan == 'esp':
        plt.legend(labels=["Media", "Mediana"])
    elif lan == 'eng':
        plt.legend(labels=["Mean", "Median"])
    
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()

##########################################################################################################

def QQplot(data, variable, title, lan, save_path=None, filename=None):
    """
    Generates a probability plot for a given variable in the dataset and saves the plot as a PNG file.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variable to analyze.
    - variable (str): The column name to check for normality.
    - title (str): The title of the plot.
    - lan (str): The language for the labels ('esp' for Spanish, 'eng' for English).
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """
    plt.figure()
    res = stats.probplot(data[variable].dropna(), dist="norm", plot=plt)
    plt.title(title)
    
    if lan == 'esp':
        plt.xlabel("Cuantiles teóricos")
        plt.ylabel("Cuantiles de la muestra")
        plt.legend(["Muestra", "Ajuste de la recta"], loc="best")
    elif lan == 'eng':
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.legend(["Sample", "Fitted Line"], loc="best")
    
    plt.grid()
    
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()
##########################################################################################################

def PPplot(data, variable, title, lan, save_path=None, filename=None):
    """
    Generates a percentile-percentile (P-P) plot to check the normality of the variable and saves it as a PNG file.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variable to analyze.
    - variable (str): The column name to check for distribution.
    - title (str): The title of the plot.
    - lan (str): The language for labels ('esp' for Spanish, 'eng' for English).
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """
    sorted_data = np.sort(data[variable].dropna())
    percentiles = np.linspace(0, 100, len(sorted_data))
    
    # Fit to a normal distribution
    mu, sigma = norm.fit(sorted_data)
    theoretical_percentiles = norm.ppf(percentiles / 100, loc=mu, scale=sigma)

    plt.figure(figsize=(8, 6))
    plt.plot(theoretical_percentiles, sorted_data, 'o', label="Data")
    plt.plot(theoretical_percentiles, theoretical_percentiles, 'r--', label="Reference Line")
    plt.title(title)
    
    if lan == 'esp':
        plt.xlabel("Percentiles teóricos")
        plt.ylabel("Percentiles de la muestra")
        plt.legend(loc="best")
    elif lan == 'eng':
        plt.xlabel("Theoretical Percentiles")
        plt.ylabel("Sample Percentiles")
        plt.legend(["Data", "Reference Line"], loc="best")
    
    plt.grid()
    
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()

##########################################################################################################

def HistModel(data, variable, bins, lan='eng', save_path=None, filename=None):
    """
    Generate a histogram and fit a normal distribution to the data, saving the plot as a PNG file.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - variable (str): Column name of the variable to analyze.
    - bins (int or str): Number of bins for the histogram.
    - lan (str): Language for labels ('eng' for English, 'esp' for Spanish). Default is 'eng'.
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """

    # Extract the specified variable from the data
    values = data[variable].dropna()

    # Fit a normal distribution to the data
    mu, std = norm.fit(values)

    # Plot the histogram
    plt.hist(values, bins=bins, density=True, alpha=0.6, color='gray', edgecolor='black')

    # Plot the fitted normal distribution (PDF)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2, label='Normal fit' if lan == 'eng' else 'Ajuste normal')

    # Set title and labels based on language
    if lan == 'esp':
        title_text = f"Resultados del ajuste: mu = {mu:.2f}, std = {std:.2f}"
        xlabel = f"{variable}"
        ylabel = "Densidad"
        legend_label = "Ajuste normal"
    else:
        title_text = f"Fit results: mu = {mu:.2f}, std = {std:.2f}"
        xlabel = f"{variable}"
        ylabel = "Density"
        legend_label = "Normal fit"

    plt.title(title_text)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([legend_label])

    # Save the plot if the save_path and filename are provided
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    # Show the plot
    plt.show()

##########################################################################################################

def CDF_fit(data, variable, lan='eng', save_path=None, filename=None):
    """
    Generate an empirical cumulative distribution function (CDF) and fit a normal distribution.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - variable (str): Column name of the variable to analyze.
    - lan (str): Language for labels ('eng' for English, 'esp' for Spanish). Default is 'eng'.
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """
    # Extract the specified variable from the data and clean it
    values = data[variable].dropna()
    values = values.replace([np.inf, -np.inf], np.nan).dropna()  # Remove infinities

    if values.empty:
        print("Error: The provided data contains only NaN or Inf values.")
        return

    # Compute the empirical CDF
    ecdf = ECDF(values)
    plt.scatter(ecdf.x, ecdf.y, label='Empirical CDF' if lan == 'eng' else 'CDF empírica', color='blue')

    # Fit a normal distribution to the data
    mu, std = norm.fit(values)

    # Plot the fitted normal CDF
    xmin, xmax = values.min(), values.max()
    x = np.linspace(xmin, xmax, 100)
    p = norm.cdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2, label='Normal CDF' if lan == 'eng' else 'CDF normal')

    # Set title and labels based on language
    if lan == 'esp':
        title_text = f"Resultados del ajuste: mu = {mu:.2f}, std = {std:.2f}"
        xlabel = f"{variable}"
        ylabel = "Frecuencia acumulada"
    else:
        title_text = f"Fit results: mu = {mu:.2f}, std = {std:.2f}"
        xlabel = f"{variable}"
        ylabel = "Cumulative frequency"

    plt.title(title_text)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()

##########################################################################################################

def KS_Test(data, variable, lan='eng'):
    """
    Performs the Kolmogorov-Smirnov test for normality on the given data.

    Parameters:
    - data (pd.DataFrame or np.ndarray): Dataset containing the variable.
    - variable (str): Name of the variable to be tested.
    - lan (str): Language for the output ('eng' for English, 'esp' for Spanish). Default is 'eng'.

    Returns:
    - None: Prints the KS test results in the specified language.
    """
    # Extract the values from the data
    values = data[variable] if isinstance(data, pd.DataFrame) else data

    # Perform the KS test
    ks_statistic, p_value = kstest(values, 'norm', args=(np.mean(values), np.std(values)))

    # Display results based on the selected language
    if lan == 'esp':
        print(f"Prueba de Kolmogorov-Smirnov para {variable}:")
        print(f"Estadístico KS: {ks_statistic:.4f}")
        print(f"Valor p: {p_value:.4f}")
        if p_value > 0.05:
            print("No hay evidencia suficiente para rechazar la normalidad.")
        else:
            print("Se rechaza la hipótesis de normalidad.")
    else:
        print(f"Kolmogorov-Smirnov test for {variable}:")
        print(f"KS Statistic: {ks_statistic:.4f}")
        print(f"P-value: {p_value:.4f}")
        if p_value > 0.05:
            print("There is not enough evidence to reject normality.")
        else:
            print("The normality hypothesis is rejected.")

##########################################################################################################

def anderson_test(data, variable, lan='eng'):
    """
    Performs the Anderson-Darling test for normality on the given data.

    Parameters:
    - data (pd.DataFrame or np.ndarray): Dataset containing the variable.
    - variable (str): Name of the variable to be tested.
    - lan (str): Language for the output ('eng' for English, 'esp' for Spanish). Default is 'eng'.

    Returns:
    - None: Prints the Anderson-Darling test results in the specified language.
    """
    # Extract the values from the data
    values = data[variable] if isinstance(data, pd.DataFrame) else data

    # Perform the Anderson-Darling test
    result = anderson(values, dist='norm')

    if lan == 'esp':
        print(f"Prueba de Anderson-Darling para {variable}:")
        print(f"Estadístico: {result.statistic:.3f}")
    else:
        print(f"Anderson-Darling test for {variable}:")
        print(f"Statistic: {result.statistic:.3f}")

    # Interpret the results based on critical values
    for i in range(len(result.critical_values)):
        slevel, cvalues = result.significance_level[i], result.critical_values[i]
        if result.statistic < cvalues:
            if lan == 'esp':
                print(f"{slevel:.3f}: {cvalues:.3f}, los datos parecen normales (no se rechaza H0)")
            else:
                print(f"{slevel:.3f}: {cvalues:.3f}, data looks normal (fail to reject H0)")
        else:
            if lan == 'esp':
                print(f"{slevel:.3f}: {cvalues:.3f}, los datos no parecen normales (se rechaza H0)")
            else:
                print(f"{slevel:.3f}: {cvalues:.3f}, data does not look normal (reject H0)")

##########################################################################################################
    
def HistBoxplotEv(data, x, hue, bins, stat, color, xlab, lan, save_path=None, filename=None):
    """
    Create a histogram and boxplot with events, calculating mean and median, and save the plot as a PNG file.

    Parameters:
    - data (pd.DataFrame): Dataset containing the columns.
    - x (str): Column name for the x variable.
    - hue (str): Column name for the event categories.
    - bins (int): Number of bins for the histogram.
    - stat (str): Type of histogram statistics (e.g., 'count', 'density').
    - color (str): Color for the boxplot.
    - xlab (str): Label for the x-axis.
    - lan (str): Language for labels ('esp' for Spanish, 'eng' for English).
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """
    mean = data[x].mean()  # Calculate mean
    median = data[x].median()  # Calculate median

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                        gridspec_kw={"height_ratios": (.15, .85)})

    sns.boxplot(data=data, x=x, orient='h', color=color, notch=True, ax=ax_box)
    line1 = ax_box.axvline(mean, ymin=0, ymax=1, color='red', linestyle="dashed")
    line2 = ax_box.axvline(median, ymin=0, ymax=1, color='blue', linestyle="dashed")

    if lan == 'esp':
        ax_box.legend(bbox_to_anchor=(0.75, 1), handles=[line1, line2], labels=["Media", "Mediana"])
    elif lan == 'eng':
        ax_box.legend(bbox_to_anchor=(0.75, 1), handles=[line1, line2], labels=["Mean", "Median"])

    sns.set_style("whitegrid", {'grid.linestyle': '-.'})
    sns.histplot(data=data, x=x, hue=hue, multiple="stack", palette="tab10", bins=bins, stat=stat, ax=ax_hist)
    ax_hist.axvline(mean, ymin=0, ymax=1, color='red', linestyle="dashed")
    ax_hist.axvline(median, ymin=0, ymax=1, color='blue', linestyle="dashed")

    plt.xlabel(xlab)
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()

##########################################################################################################

def BarBoxplotEv(data, x, hue, bins, stat, color, xlab, lan, save_path=None, filename=None):
    """
    Create a barplot and boxplot with events, calculating mean and median, and save the plot as a PNG file.

    Parameters:
    - data (pd.DataFrame): Dataset containing the columns.
    - x (str): Column name for the x variable.
    - hue (str): Column name for the event categories.
    - bins (int): Number of bins for the histogram.
    - stat (str): Type of histogram statistics (e.g., 'count', 'density').
    - color (str): Color for the boxplot.
    - xlab (str): Label for the x-axis.
    - lan (str): Language for labels ('esp' for Spanish, 'eng' for English).
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """
    mean = data[x].mean()  # Calculate mean
    median = data[x].median()  # Calculate median

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                        gridspec_kw={"height_ratios": (.25, .75)})

    sns.boxplot(data=data, x=x, hue=hue, orient='h', palette="tab10", notch=True, ax=ax_box)
    line1 = ax_box.axvline(mean, ymin=0, ymax=1, color='red', linestyle="dashed")
    line2 = ax_box.axvline(median, ymin=0, ymax=1, color='blue', linestyle="dashed")

    if lan == 'esp':
        ax_box.legend(bbox_to_anchor=(0.75, 1), handles=[line1, line2], labels=["Media", "Mediana"])
    elif lan == 'eng':
        ax_box.legend(bbox_to_anchor=(0.75, 1), handles=[line1, line2], labels=["Mean", "Median"])

    sns.set_style("whitegrid", {'grid.linestyle': '-.'})
    sns.histplot(data=data, x=x, hue=hue, multiple="dodge", palette="tab10", bins=bins, stat=stat, ax=ax_hist)
    ax_hist.axvline(mean, ymin=0, ymax=1, color='red', linestyle="dashed")
    ax_hist.axvline(median, ymin=0, ymax=1, color='blue', linestyle="dashed")

    plt.xlabel(xlab)
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()

##########################################################################################################

def OutliersPos(data, variable, lan='esp'):
    """
    Identifies outliers in the specified column of the dataset based on the IQR method.

    Parameters:
    - data: pd.DataFrame, the dataset containing the column to analyze.
    - variable: str, the column name to check for outliers.
    - lan: str, the language for the output ('esp' for Spanish, 'eng' for English).
    - allow_negative: bool, whether to allow negative values in lower bound.

    Returns:
    - outliers: pd.DataFrame containing the rows with outliers or a message if no outliers are found.
    """

    # Calculate the quartiles and IQR
    Q1 = data[variable].quantile(0.25)
    Q3 = data[variable].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data[variable] < lim_inf) | (data[variable] > lim_sup)]

    # Return results
    if outliers.empty:
        return "No hay valores atípicos." if lan == 'esp' else "There are no outliers."
    
    return outliers

##########################################################################################################

def RemoveOutliersPos(data, variable, save_path, filename, lan='esp'):
    """
    Removes outliers from the specified variable based on the IQR method and saves the cleaned data to a CSV file.

    Parameters:
    - data: pd.DataFrame, the dataset containing the variable to analyze.
    - variable: str, the column name to check for outliers.
    - save_path (str): Directory path to save the cleaned dataset.
    - filename (str): Filename to save the cleaned data as CSV.
    - lan: str, the language for the output ('esp' for Spanish, 'eng' for English).

    Returns:
    - pd.DataFrame containing the rows without outliers.
    """
    # Calculate the quartiles and IQR
    Q1 = data[variable].quantile(0.25)
    Q3 = data[variable].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    
    # Identify rows without outliers
    var = data[variable]
    non_outliers = data[(var >= lim_inf) & (var <= lim_sup)]
    
    # Reset index for the cleaned DataFrame
    non_outliers.reset_index(drop=True, inplace=True)

    # Save the cleaned data to a CSV file
    file_path = os.path.join(save_path, f"{filename}.csv")
    non_outliers.to_csv(file_path, index=False)

    return non_outliers

##########################################################################################################

def plot_event_based_median_regression(data, x, y, events, save_path=None):
    """
    Plot median regression for each event with event-specific statistics in a 1-row, 3-column layout.

    Parameters:
    - data (pd.DataFrame): The dataset containing the columns.
    - x (str): Column name for the independent variable (e.g., spatial coordinate, time).
    - y (str): Column name for the dependent variable.
    - events (str): Column name for event divisions.

    Returns:
    - None
    """
    # Convert x to numeric if it's a datetime object
    data[x] = pd.to_datetime(data[x])  # Convert x to datetime
    data[f"{x}_numeric"] = (data[x] - data[x].iloc[0]).dt.total_seconds()  # Convert to numeric

    unique_events = sorted(data[events].unique())
    num_events = len(unique_events)

    fig, axes = plt.subplots(1, num_events, figsize=(6 * num_events, 6), sharey=True)

    if num_events == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    for i, event in enumerate(unique_events):
        ax = axes[i]

        # Filter data for the event
        event_data = data[data[events] == event]
        X_event = event_data[[f"{x}_numeric"]]
        Y_event = event_data[y]

        # Median regression model
        median_model = QuantileRegressor(quantile=0.5, alpha=0, solver='highs').fit(X_event, Y_event)
        event_data['Median_Regression'] = median_model.predict(X_event)

        # Calculate statistics
        q1 = Y_event.quantile(0.25)  # 1st quartile
        median = Y_event.median()    # Median
        mean_value = Y_event.mean()  # Mean
        q3 = Y_event.quantile(0.75)  # 3rd quartile

        # Plot data as a line
        ax.plot(event_data[x], Y_event, color='black', alpha=0.7, label=f"Event {event} Data")

        # Plot median regression
        ax.plot(event_data[x], event_data['Median_Regression'], color='blue', linestyle=':', label="Median Regression")

        # Add horizontal lines for statistics
        ax.axhline(q1, color='magenta', linestyle='-', label="1st Quartile")
        ax.axhline(mean_value, color='red', linestyle='-', label="Mean")
        ax.axhline(median, color='blue', linestyle='-', label="Median")
        ax.axhline(q3, color='cyan', linestyle='-', label="3rd Quartile")

        # Customize subplot
        ax.set_title(f"Event {event}")
        ax.set_xlabel(x)
        if i == 0:
            ax.set_ylabel(y)
        ax.legend(loc="best")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

##########################################################################################################

def plot_median_regression_with_events(data, x, y, events, save_path=None):
    """
    Plot median regression for all data with event divisions, horizontal lines for statistics, and a line plot.

    Parameters:
    - data (pd.DataFrame): The dataset containing the columns.
    - x (str): Column name for the independent variable (e.g., spatial coordinate, time).
    - y (str): Column name for the dependent variable.
    - events (str): Column name for event divisions.

    Returns:
    - None
    """
    # Ensure x is datetime and convert to numeric
    data[x] = pd.to_datetime(data[x])  # Convert x to datetime
    data[f"{x}_numeric"] = (data[x] - data[x].iloc[0]).dt.total_seconds()  # Convert to numeric

    # Use the numeric version of x for regression
    X = data[[f"{x}_numeric"]]
    Y = data[y]

    # Median regression model
    median_model = QuantileRegressor(quantile=0.5, alpha=0, solver='highs').fit(X, Y)
    data['Median_Regression'] = median_model.predict(X)

    # Calculate statistics
    q1 = Y.quantile(0.25)  # 1st quartile
    median = Y.median()    # Median
    mean_value = Y.mean()  # Mean
    q3 = Y.quantile(0.75)  # 3rd quartile

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(data[x], Y, color='black', alpha=0.7, label=f"{y} Data")  # Line plot

    # Plot median regression
    ax.plot(data[x], data['Median_Regression'], color='blue', linestyle=':', label="Median Regression")

    # Add horizontal lines for statistics
    ax.axhline(q1, color='magenta', linestyle='-', label="1st Quartile")
    ax.axhline(mean_value, color='red', linestyle='-', label="Mean")
    ax.axhline(median, color='blue', linestyle='-', label="Median")
    ax.axhline(q3, color='cyan', linestyle='-', label="3rd Quartile")

    # Event shading
    unique_events = data[events].unique()
    palette = {0: 'steelblue', 1: 'darkorange', 2: 'forestgreen'}
    for event in unique_events:
        if event in palette:
            event_indices = data[data[events] == event].index
            start_time = data.loc[event_indices[0], x]
            end_time = data.loc[event_indices[-1], x]
            ax.axvspan(start_time, end_time, color=palette[event], alpha=0.3, label=f"Event {event}")

    # Customize plot
    ax.set_title(f"Median Regression and Events for {y} vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

##########################################################################################################

def ScatterPlot(data, x, y, bins, xlab, ylab, lan='eng', save_path=None, filename=None):
    """
    Python equivalent of the R ScatterPlot function with scatter plot, histograms, boxplots, and statistical measures.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - x (str): Column name for the x-axis variable.
    - y (str): Column name for the y-axis variable.
    - bins (int): Number of bins for histograms.
    - xlab (str): Label for x-axis.
    - ylab (str): Label for y-axis.
    - lan (str): Language for labels ('eng' for English, 'esp' for Spanish). Default is 'eng'.
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """
    # Extract x and y columns
    x_data = data[x]
    y_data = data[y]

    # Calculate axis limits
    xmin, xmax = x_data.min(), x_data.max()
    ymin, ymax = y_data.min(), y_data.max()

    # Adjust axis limits dynamically
    xmin = xmin * 1.1 if xmin < 0 else xmin * 0.9
    xmax = xmax * 0.9 if xmax < 0 else xmax * 1.1
    ymin = ymin * 1.1 if ymin < 0 else ymin * 0.9
    ymax = ymax * 0.9 if ymax < 0 else ymax * 1.1

    # Compute correlation coefficients
    pearson_corr = x_data.corr(y_data, method='pearson')
    spearman_corr = x_data.corr(y_data, method='spearman')
    kendall_corr = x_data.corr(y_data, method='kendall')

    # Create a grid layout with reduced spacing
    fig = plt.figure(figsize=(17, 12))
    grid = plt.GridSpec(6, 6, hspace=0.2, wspace=0.2)  # Reduced hspace and wspace

    # Scatter plot
    ax_main = fig.add_subplot(grid[2:, 1:4])
    ax_main.scatter(x_data, y_data, color='black', s=10)
    ax_main.set_xlim([xmin, xmax])
    ax_main.set_ylim([ymin, ymax])
    ax_main.set_xlabel(xlab)
    ax_main.set_ylabel(ylab)
    ax_main.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Top histogram (X)
    ax_hist_x = fig.add_subplot(grid[0, 1:4])
    sns.histplot(x_data, bins=bins, color='#00AAFF', kde=False, ax=ax_hist_x)
    ax_hist_x.set_ylabel('Frequency')
    ax_hist_x.grid(False)
    plt.setp(ax_hist_x.get_xticklabels(), visible=False)

    # Horizontal boxplot for X
    ax_box_x = fig.add_subplot(grid[1, 1:4])
    sns.boxplot(x=x_data, ax=ax_box_x, color='#00AAFF', orient='h', notch=True, width=0.5)
    ax_box_x.axis('on')
    plt.setp(ax_box_x.get_xticklabels(), visible=False)
    sns.despine(ax=ax_box_x, left=True, bottom=True)

    # Right histogram (Y)
    ax_hist_y = fig.add_subplot(grid[2:, 5])
    bin_counts, bin_edges = np.histogram(y_data, bins=bins)
    ax_hist_y.barh(bin_edges[:-1], bin_counts, height=np.diff(bin_edges), color='#CCFF00', align='edge')
    ax_hist_y.set_ylim([ymin, ymax])
    ax_hist_y.set_xlim([0, max(bin_counts)])
    ax_hist_y.set_xlabel('Frequency')
    ax_hist_y.grid(False)

    # Vertical boxplot for Y
    ax_box_y = fig.add_subplot(grid[2:, 4])
    sns.boxplot(y=y_data, ax=ax_box_y, color='#CCFF00', orient='v', notch=True, width=0.5)
    ax_box_y.set_ylim([ymin, ymax])
    ax_box_y.invert_xaxis()
    ax_box_y.axis('on')
    plt.setp(ax_box_y.get_yticklabels(), visible=False)
    sns.despine(ax=ax_box_y, left=True, bottom=True)

    # Correlation text
    ax_corr = fig.add_subplot(grid[0, 4:])
    ax_corr.axis('off')
    if lan == 'esp':
        title_text = "Medidas de Dependencia"
        corr_text = (f"{title_text}\n"
                     f"Pearson = {pearson_corr:.4f}\n"
                     f"Spearman = {spearman_corr:.4f}\n"
                     f"Kendall = {kendall_corr:.4f}")
    else:
        title_text = "Dependence Measures"
        corr_text = (f"{title_text}\n"
                     f"Pearson = {pearson_corr:.4f}\n"
                     f"Spearman = {spearman_corr:.4f}\n"
                     f"Kendall = {kendall_corr:.4f}")
    ax_corr.text(0.5, 0.5, corr_text, ha='center', va='center', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='black'))

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    # Show the plot
    plt.show()

##########################################################################################################

def ScatterPlotCat(data, x, y, bins, xlab, ylab, categorical_var, lan='eng', save_path=None, filename=None):
    """
    Generates a categorical scatter plot with histograms, boxplots, and statistical measures,
    and saves the plot as a PNG file.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - x (str): Column name for the x-axis variable.
    - y (str): Column name for the y-axis variable.
    - bins (int or str): Number of bins for histograms.
    - xlab (str): Label for x-axis.
    - ylab (str): Label for y-axis.
    - categorical_var (str): Column name of the categorical variable.
    - lan (str): Language for labels ('eng' or 'esp'). Default is 'eng'.
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """
    if categorical_var not in data.columns:
        raise ValueError(f"Categorical variable '{categorical_var}' not found in dataset.")

    # Calculate axis limits dynamically
    xmin, xmax = data[x].min(), data[x].max()
    ymin, ymax = data[y].min(), data[y].max()

    # Adjust limits
    xmin = xmin * 1.1 if xmin < 0 else xmin * 0.9
    xmax = xmax * 0.9 if xmax < 0 else xmax * 1.1
    ymin = ymin * 1.1 if ymin < 0 else ymin * 0.9
    ymax = ymax * 0.9 if ymax < 0 else ymax * 1.1

    # Compute correlation coefficients
    pearson_corr = data[x].corr(data[y], method='pearson')
    spearman_corr = data[x].corr(data[y], method='spearman')
    kendall_corr = data[x].corr(data[y], method='kendall')

    # Create the plot
    fig = plt.figure(figsize=(17, 12))
    grid = plt.GridSpec(6, 6, hspace=0.2, wspace=0.2)

    # Scatter plot with categorical hue
    ax_main = fig.add_subplot(grid[2:, 1:4])
    sns.scatterplot(x=data[x], y=data[y], hue=data[categorical_var], ax=ax_main, palette="tab10")
    ax_main.set_xlim([xmin, xmax])
    ax_main.set_ylim([ymin, ymax])
    ax_main.set_xlabel(xlab)
    ax_main.set_ylabel(ylab)
    ax_main.grid(color='lightgray', linestyle='--', linewidth=0.5)
    ax_main.legend(title=categorical_var, loc='best')

    # Histograms with categories
    ax_hist_x = fig.add_subplot(grid[0, 1:4])
    sns.histplot(data, x=x, hue=categorical_var, bins=bins, palette="tab10", ax=ax_hist_x)
    ax_hist_x.set_ylabel('Frequency')
    ax_hist_x.grid(False)
    plt.setp(ax_hist_x.get_xticklabels(), visible=False)

    # Boxplots for X variable categorized
    ax_box_x = fig.add_subplot(grid[1, 1:4])
    sns.boxplot(x=x, y=categorical_var, data=data, palette="tab10", ax=ax_box_x, orient='h', notch=True)
    plt.setp(ax_box_x.get_xticklabels(), visible=False)

    # Right-side histogram
    ax_hist_y = fig.add_subplot(grid[2:, 5])
    sns.histplot(data, y=y, hue=categorical_var, bins=bins, palette="tab10", ax=ax_hist_y)
    ax_hist_y.set_xlabel('Frequency')
    ax_hist_y.grid(False)

    # Vertical boxplot categorized
    ax_box_y = fig.add_subplot(grid[2:, 4])
    sns.boxplot(y=y, x=categorical_var, data=data, palette="tab10", ax=ax_box_y, notch=True)
    ax_box_y.set_ylim([ymin, ymax])
    ax_box_y.invert_xaxis()
    plt.setp(ax_box_y.get_yticklabels(), visible=False)

    # Correlation text
    ax_corr = fig.add_subplot(grid[0, 4:])
    ax_corr.axis('off')
    if lan == 'esp':
        title_text = "Medidas de Dependencia"
        corr_text = (f"{title_text}\n"
                     f"Pearson = {pearson_corr:.4f}\n"
                     f"Spearman = {spearman_corr:.4f}\n"
                     f"Kendall = {kendall_corr:.4f}")
    else:
        title_text = "Dependence Measures"
        corr_text = (f"{title_text}\n"
                     f"Pearson = {pearson_corr:.4f}\n"
                     f"Spearman = {spearman_corr:.4f}\n"
                     f"Kendall = {kendall_corr:.4f}")

    ax_corr.text(0.5, 0.5, corr_text, ha='center', va='center', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='black'))

    # Title based on language
    main_title = "Categorical Scatter Plot" if lan == 'eng' else "Diagrama de Dispersión Categórico"
    plt.suptitle(main_title, fontsize=14, fontweight='bold')

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()

##########################################################################################################

def dependency_matrix(data, variables, method='pearson', output='image', filename='dependency_matrix', lan='eng', save_path=None):
    """
    Computes and visualizes the dependency matrix between variables.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing the variables of interest.
    variables : list
        List of column names to analyze.
    method : str, optional
        Correlation method ('pearson', 'spearman', or 'kendall'). Default is 'pearson'.
    output : str, optional
        Output type: 'image' to display/save the heatmap or 'table' to return/save the correlation matrix.
    filename : str, optional
        Filename (without extension) to save the dependency matrix. Default is 'dependency_matrix'.
    lan : str, optional
        Language for titles and labels ('esp' for Spanish, 'eng' for English). Default is 'eng'.
    save_path : str, optional
        Directory to save the plot or table file. Default is None.

    Returns:
    --------
    If `output` is 'image', displays the dependency matrix as a heatmap.
    If `output` is 'table', returns a DataFrame with the correlation matrix and optionally saves it as a CSV file.
    """
    
    # Filter the variables of interest
    filtered_data = data[variables]
    
    # Compute the correlation matrix
    correlation_matrix = filtered_data.corr(method=method)
    
    # Titles based on the selected language
    titles = {
        'eng': f'Dependency Matrix ({method.capitalize()} correlation)',
        'esp': f'Matriz de Dependencia (Correlación de {method.capitalize()})'
    }
    
    # Labels based on the selected language
    labels = {
        'eng': {'xlabel': 'Variables', 'ylabel': 'Variables'},
        'esp': {'xlabel': 'Variables', 'ylabel': 'Variables'}
    }

    if output == 'image':
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, cmap='seismic', fmt=".2f")
        plt.title(titles[lan])
        plt.xlabel(labels[lan]['xlabel'])
        plt.ylabel(labels[lan]['ylabel'])
        
        # Handle the save_path if provided
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)  # Create the directory if it doesn't exist
            filename = os.path.join(save_path, f"{filename}.png")
        else:
            filename = f"{filename}.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    elif output == 'table':
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            csv_filename = os.path.join(save_path, f"{filename}.csv")
        else:
            csv_filename = f"{filename}.csv"

        # Save the correlation matrix to a CSV file
        correlation_matrix.to_csv(csv_filename, index=True)
        print(f"Correlation matrix saved as CSV at: {csv_filename}")
        return correlation_matrix
    
    else:
        raise ValueError("The 'output' parameter must be 'image' or 'table'.")

##########################################################################################################

def scatterplot_matrix(data, variables, categorical_var=None, filename='scatterplot_matrix.png', lan='eng', save_path=None):
    """
    Generates a scatterplot matrix for the given variables in the dataset.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variables of interest.
    - variables (list): List of column names to include in the scatterplot matrix.
    - categorical_var (str, optional): The name of the categorical variable for coloring the scatterplot. Default is None.
    - filename (str, optional): The name of the file to save the scatterplot matrix. Default is 'scatterplot_matrix.png'.
    - lan (str, optional): Language for the title and legends ('esp' for Spanish, 'eng' for English). Default is 'eng'.
    - save_path (str, optional): Directory to save the scatterplot matrix.

    Returns:
    - None
    """
    titles = {'eng': 'Scatterplot Matrix', 'esp': 'Matriz de Diagramas de Dispersión'}

    if categorical_var and categorical_var in data.columns:
        plot = sns.pairplot(data, vars=variables, hue=categorical_var, diag_kind='kde', palette="tab10")
        plot.fig.suptitle(titles[lan], fontsize=16, y=1.05)
        plot._legend.set_bbox_to_anchor((1.05, 0.5))
    else:
        plot = sns.pairplot(data, vars=variables, diag_kind='kde')
        plot.fig.suptitle(titles[lan], fontsize=16, y=1.05)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, filename)
    plot.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

##########################################################################################################    

def mahalanobis_distance_plot2D(data, variables, save_path=None, filename='mahalanobis_plot.png', lan='eng'):

    """
    Computes Mahalanobis distance for two variables, identifies outliers using the boxplot rule,
    and returns the cleaned DataFrame without outliers.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables of interest.
    - variables (list): List of two column names for Mahalanobis distance calculation.
    - save_path (str, optional): Directory to save the plot. If None, the plot is displayed but not saved.
    - filename (str, optional): Filename for saving the plot. Default is 'mahalanobis_plot.png'.
    - lan (str, optional): Language for plot labels ('esp' for Spanish, 'eng' for English). Default is 'eng'.

    Returns:
    - pd.DataFrame: DataFrame of variables without outliers.
    """
    df = data[variables].dropna()

    # Compute Mahalanobis distance
    mean_vector = np.mean(df, axis=0)
    cov_matrix = np.cov(df.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_distances = df.apply(lambda row: mahalanobis(row, mean_vector, inv_cov_matrix), axis=1)
    df['Mahalanobis_Dist'] = mahalanobis_distances

    # Identify outliers using boxplot rule
    Q1 = df['Mahalanobis_Dist'].quantile(0.25)
    Q3 = df['Mahalanobis_Dist'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR, df['Mahalanobis_Dist'].min())
    upper_bound = Q3 + 1.5 * IQR

    df['Outlier'] = (df['Mahalanobis_Dist'] < lower_bound) | (df['Mahalanobis_Dist'] > upper_bound)

    # Language settings for labels
    lang_labels = {
        'en': {'title': 'Histogram of Distances', 'xlabel': 'Mahalanobis Distance', 'ylabel': 'Frequency', 
               'lower': 'Lower Bound', 'upper': 'Upper Bound', 'outlier': 'Outlier', 'no': 'No', 'yes': 'Yes'},
        'esp': {'title': 'Histograma de Distancias', 'xlabel': 'Distancia de Mahalanobis', 'ylabel': 'Frecuencia', 
                'lower': 'Límite Inferior', 'upper': 'Límite Superior', 'outlier': 'Atípico', 'no': 'No', 'yes': 'Sí'}
    }

    labels = lang_labels.get(lan, lang_labels['en'])

    # Create figure with subplots (boxplot above histogram on left, scatter plot on right)
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.3, 0.7], width_ratios=[1, 1])

    # Boxplot (top-left)
    ax_box = fig.add_subplot(gs[0, 0])
    sns.boxplot(x=df['Mahalanobis_Dist'], color='skyblue', ax=ax_box, notch=True)
    ax_box.axvline(lower_bound, color='red', linestyle='--', label=labels['lower'])
    ax_box.axvline(upper_bound, color='red', linestyle='--', label=labels['upper'])
    ax_box.set_xlabel('')
    ax_box.set_yticks([])

    # Histogram (bottom-left)
    ax_hist = fig.add_subplot(gs[1, 0])
    sns.histplot(df['Mahalanobis_Dist'], bins=30, kde=True, ax=ax_hist, color='skyblue', alpha=0.6)
    ax_hist.axvline(lower_bound, color='red', linestyle='--', label=labels['lower'])
    ax_hist.axvline(upper_bound, color='red', linestyle='--', label=labels['upper'])
    ax_hist.set_title(labels['title'])
    ax_hist.set_xlabel(labels['xlabel'])
    ax_hist.set_ylabel(labels['ylabel'])
    ax_hist.legend()

    # Scatter plot (right side)
    ax_scatter = fig.add_subplot(gs[:, 1])
    scatter = sns.scatterplot(
        x=df[variables[0]],
        y=df[variables[1]],
        hue=df['Outlier'].map({False: labels['no'], True: labels['yes']}),
        palette={labels['no']: 'blue', labels['yes']: 'red'},
        edgecolor='k',
        s=60,
        ax=ax_scatter
    )
    num_outliers = df['Outlier'].sum()
    ax_scatter.set_title(f"{labels['outlier']} ({num_outliers} outliers)")
    ax_scatter.set_xlabel(variables[0])
    ax_scatter.set_ylabel(variables[1])
    
    # Fix legend issue by manually creating it
    handles, _ = scatter.get_legend_handles_labels()
    ax_scatter.legend(handles, [labels['no'], labels['yes']], title=labels['outlier'])

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_filepath = os.path.join(save_path, filename)
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_filepath}")
    else:
        plt.show()

    # Return DataFrame without outliers
    return df[~df['Outlier']].drop(columns=['Mahalanobis_Dist', 'Outlier'])

##########################################################################################################

def mahalanobis_distance_outlier_removal(data, variables, file_name, lan, save_path):
    """
    Calculate Mahalanobis distance, identify outliers, and save histogram with boxplot and scatterplot matrix.

    Parameters:
    data (pd.DataFrame): The input dataset.
    variables (list): List of variable names to consider.
    file_name (str): Name of the saved plot.
    lan (str): Language option for labeling ('en' or 'es').
    save_path (str): Path to save the image.

    Returns:
    pd.DataFrame: Cleaned data without outliers.
    """
    # Select relevant data
    df_selected = data[variables]

    # Compute Mahalanobis distance
    cov_matrix = np.cov(df_selected.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean_vector = np.mean(df_selected, axis=0)

    mahal_dist = df_selected.apply(lambda row: mahalanobis(row, mean_vector, inv_cov_matrix), axis=1)
    df_selected['Mahalanobis_Dist'] = mahal_dist

    # Identify outliers based on boxplot rule
    Q1 = df_selected['Mahalanobis_Dist'].quantile(0.25)
    Q3 = df_selected['Mahalanobis_Dist'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR, df_selected['Mahalanobis_Dist'].min())
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df_selected[(df_selected['Mahalanobis_Dist'] >= lower_bound) & (df_selected['Mahalanobis_Dist'] <= upper_bound)].drop(columns=['Mahalanobis_Dist'])
    df_outliers = df_selected[(df_selected['Mahalanobis_Dist'] < lower_bound) | (df_selected['Mahalanobis_Dist'] > upper_bound)].drop(columns=['Mahalanobis_Dist'])

    # Create a new column to indicate outliers
    df_selected['Outlier'] = df_selected.index.isin(df_outliers.index)

    # Create the figure layout with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 3]})

    # Boxplot for Mahalanobis distance
    ax_box = axes[0]
    sns.boxplot(x=df_selected['Mahalanobis_Dist'], ax=ax_box, color='skyblue', notch=True)
    ax_box.axvline(lower_bound, color='red', linestyle='--')
    ax_box.axvline(upper_bound, color='red', linestyle='--')
    ax_box.set_title('Boxplot of Mahalanobis Distance' if lan == 'en' else 'Diagrama de Caja de la Distancia Mahalanobis')
    ax_box.set_xlabel('Mahalanobis Distance')

    # Histogram with Mahalanobis distance
    ax_hist = axes[1]
    sns.histplot(df_selected['Mahalanobis_Dist'], kde=False, color='skyblue', ax=ax_hist)
    ax_hist.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
    ax_hist.axvline(upper_bound, color='red', linestyle='--', label='Upper Bound')
    ax_hist.set_title('Mahalanobis Distance Histogram' if lan == 'en' else 'Histograma de Distancia Mahalanobis')
    ax_hist.set_xlabel('Mahalanobis Distance')
    ax_hist.set_ylabel('Frequency')
    ax_hist.legend()



    # Save the histogram with boxplot
    hist_path = f"{save_path}/{file_name}_histogram_boxplot.png"
    fig.tight_layout()
    fig.savefig(hist_path, bbox_inches='tight')

    # Scatterplot matrix
    scatterplot_path = f"{save_path}/{file_name}_scatterplot_matrix.png"
    pairplot = sns.pairplot(df_selected.drop(columns=['Mahalanobis_Dist']), hue='Outlier', palette={True: 'red', False: 'blue'}, diag_kind='hist', corner=False)
    pairplot.fig.subplots_adjust(top=0.95)
    num_outliers = df_selected['Outlier'].sum()
    pairplot.fig.suptitle(
        'Scatterplot Matrix with Outliers Highlighted' if lan == 'en' else 'Matriz de Dispersión con Valores Atípicos Resaltados',
        size=14
    )
    pairplot.savefig(scatterplot_path)

    plt.close()

    return df_clean

##########################################################################################################    
    
def ScatterPlotReg(sample, x, y, bins, xlab, ylab, lan='eng', save_path=None, filename=None):
    """
    Python equivalent of the R ScatterPlot function with scatter plot, histograms, boxplots,
    linear regression, and statistical measures. The plot can be saved as a PNG file.

    Parameters:
    - sample (pd.DataFrame): Dataset containing the variables.
    - x (str): Column name for the x-axis variable.
    - y (str): Column name for the y-axis variable.
    - bins (int): Number of bins for histograms.
    - xlab (str): Label for x-axis.
    - ylab (str): Label for y-axis.
    - lan (str): Language option ('eng' for English, 'esp' for Spanish). Default is 'eng'.
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the image as PNG.

    Returns:
    - None
    """
    # Extract x and y columns
    x_data = sample[x]
    y_data = sample[y]

    # Calculate axis limits
    xmin, xmax = x_data.min(), x_data.max()
    ymin, ymax = y_data.min(), y_data.max()

    # Adjust axis limits dynamically
    xmin = xmin * 1.1 if xmin < 0 else xmin * 0.9
    xmax = xmax * 0.9 if xmax < 0 else xmax * 1.1
    ymin = ymin * 1.1 if ymin < 0 else ymin * 0.9
    ymax = ymax * 0.9 if ymax < 0 else ymax * 1.1

    # Compute correlation coefficients
    pearson_corr = x_data.corr(y_data, method='pearson')
    spearman_corr = x_data.corr(y_data, method='spearman')
    kendall_corr = x_data.corr(y_data, method='kendall')

    # Perform Linear Regression
    x_reshaped = x_data.values.reshape(-1, 1)
    y_reshaped = y_data.values
    lin_reg = LinearRegression()
    lin_reg.fit(x_reshaped, y_reshaped)

    # Predicted values and R²
    y_pred = lin_reg.predict(x_reshaped)
    r2 = r2_score(y_reshaped, y_pred)
    slope = lin_reg.coef_[0]
    intercept = lin_reg.intercept_

    # Create a grid layout with reduced spacing
    fig = plt.figure(figsize=(17, 12))
    grid = plt.GridSpec(6, 6, hspace=0.3, wspace=0.3)

    # Scatter plot with linear regression
    ax_main = fig.add_subplot(grid[2:, 1:4])
    ax_main.scatter(x_data, y_data, color='black', s=10, label='Sample' if lan == 'eng' else 'Muestra')
    ax_main.plot(x_data, y_pred, color='red', linestyle='--', label='Linear Regression' if lan == 'eng' else 'Regresión Lineal')
    ax_main.set_xlim([xmin, xmax])
    ax_main.set_ylim([ymin, ymax])
    ax_main.set_xlabel(xlab)
    ax_main.set_ylabel(ylab)
    ax_main.legend(loc="best")
    ax_main.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Top histogram (X)
    ax_hist_x = fig.add_subplot(grid[0, 1:4])
    sns.histplot(x_data, bins=bins, color='#00AAFF', kde=False, ax=ax_hist_x)
    ax_hist_x.set_ylabel('Frequency' if lan == 'eng' else 'Frecuencia')
    ax_hist_x.grid(False)
    plt.setp(ax_hist_x.get_xticklabels(), visible=False)

    # Horizontal boxplot for X
    ax_box_x = fig.add_subplot(grid[1, 1:4])
    sns.boxplot(x=x_data, ax=ax_box_x, color='#00AAFF', orient='h', notch=True, width=0.25)
    plt.setp(ax_box_x.get_xticklabels(), visible=False)
    sns.despine(ax=ax_box_x, left=True, bottom=True)

    # Right histogram (Y)
    ax_hist_y = fig.add_subplot(grid[2:, 5])
    bin_counts, bin_edges = np.histogram(y_data, bins=bins)
    ax_hist_y.barh(bin_edges[:-1], bin_counts, height=np.diff(bin_edges), color='#CCFF00', align='edge')
    ax_hist_y.set_ylim([ymin, ymax])
    ax_hist_y.set_xlabel('Frequency' if lan == 'eng' else 'Frecuencia')
    ax_hist_y.grid(False)

    # Vertical boxplot for Y
    ax_box_y = fig.add_subplot(grid[2:, 4])
    sns.boxplot(y=y_data, ax=ax_box_y, color='#CCFF00', orient='v', notch=True, width=0.25)
    ax_box_y.set_ylim([ymin, ymax])
    ax_box_y.invert_xaxis()
    plt.setp(ax_box_y.get_yticklabels(), visible=False)
    sns.despine(ax=ax_box_y, left=True, bottom=True)

    # Correlation text and regression parameters
    ax_corr = fig.add_subplot(grid[0, 4:])
    ax_corr.axis('off')
    title_text = "Dependence Measures" if lan == 'eng' else "Medidas de Dependencia"
    corr_text = (f"{title_text}\n"
                 f"Pearson = {pearson_corr:.4f}\n"
                 f"Spearman = {spearman_corr:.4f}\n"
                 f"Kendall = {kendall_corr:.4f}\n\n"
                 f"{'Regression Parameters' if lan == 'eng' else 'Parámetros de la Regresión'}\n"
                 f"Slope = {slope:.4f}\n"
                 f"Intercept = {intercept:.4f}\n"
                 f"R² = {r2:.4f}")

    ax_corr.text(0.5, 0.5, corr_text, ha='center', va='center', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='black'))

    # Set main title
    plt.suptitle("Scatter Plot with Regression" if lan == 'eng' else "Diagrama de Dispersión con Regresión",
                 fontsize=14, fontweight='bold')

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()

##############################################################################
 
def perform_pca(data, features, n_components, categorical_variable, save_path=None, filename=None):
    """
    Perform PCA on the given dataset with specified features, number of components,
    and categorical variable for coloring. Saves the scatter plot and tables as files.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - features (list): List of feature column names to include in PCA.
    - n_components (int): Number of principal components to calculate.
    - categorical_variable (str): Column name for the categorical variable.
    - save_path (str, optional): Directory path to save the output files.
    - filename (str, optional): Filename for saving the outputs.

    Returns:
    - explained_variance_df (pd.DataFrame): Explained variance ratio by PCA components.
    - loadings_df.T (pd.DataFrame): Loadings for each feature for all principal components.
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    feature_data = data[features].dropna()
    principal_components = pca.fit_transform(feature_data)
    explained_variance = pca.explained_variance_ratio_
    
    # Create PCA DataFrame
    pca_columns = [f'PC{i + 1}' for i in range(n_components)]
    pca_df = pd.DataFrame(principal_components, columns=pca_columns)
    pca_df[categorical_variable] = data[categorical_variable].reset_index(drop=True)
    
    # Determine unique categories in the categorical variable
    categories = sorted(pca_df[categorical_variable].unique())
    
    # Use tab10 color palette
    palette = sns.color_palette("tab10", n_colors=len(categories))
    color_mapping = {category: palette[i] for i, category in enumerate(categories)}
    
    # Plot scatterplots for the lower triangle only
    fig, axes = plt.subplots(n_components - 1, n_components - 1, figsize=(16, 16), constrained_layout=True)
    for i in range(1, n_components):  # Start from PC2
        for j in range(i):  # Only plot where i > j
            ax = axes[i - 1, j]
            for category in categories:
                subset = pca_df[pca_df[categorical_variable] == category]
                ax.scatter(
                    subset[f'PC{j + 1}'], subset[f'PC{i + 1}'], 
                    label=f'{categorical_variable} {category}', 
                    color=color_mapping[category], alpha=0.7
                )
            ax.set_xlabel(f'PC{j + 1}')
            ax.set_ylabel(f'PC{i + 1}')
            ax.set_title(f'PC{j + 1} vs PC{i + 1}')
    
    # Hide unused subplots
    for i in range(n_components - 1):
        for j in range(i + 1, n_components - 1):
            fig.delaxes(axes[i, j])

    # Add legend below the matrix
    handles = [
        plt.Line2D([0], [0], color=color_mapping[category], marker='o', linestyle='', label=f'{categorical_variable} {category}')
        for category in categories
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(categories), fontsize=10)

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        plot_file_path = os.path.join(save_path, f"{filename}_pca_plot.png")
        plt.savefig(plot_file_path, bbox_inches='tight')

    plt.show()
    
    # Explained variance table
    explained_variance_df = pd.DataFrame({
        'PCA Component': pca_columns,
        'Explained Variance Ratio': explained_variance
    })

    # PCA components table (loadings)
    loadings = pca.components_
    loadings_df = pd.DataFrame(loadings, columns=features, index=pca_columns)

    # Save tables as CSV if save_path and filename are provided
    if save_path and filename:
        csv_file_path = os.path.join(save_path, f"{filename}_explained_variance.csv")
        explained_variance_df.to_csv(csv_file_path, index=False)

        loadings_csv_path = os.path.join(save_path, f"{filename}_loadings.csv")
        loadings_df.T.to_csv(loadings_csv_path)

    # Print explained variance
    print("\nExplained Variance by PCA components:")
    print(explained_variance_df)
    
    # Print PCA components (loadings)
    print("\nPCA Components (Loadings):")
    print(loadings_df)
    
    return explained_variance_df, loadings_df.T

####################################################################################

def plot_pca_vectors(data, features, n_components, save_path=None, filename=None):
    """
    Plot PCA feature vectors (loadings) for the specified features and number of components,
    and save the plot and loadings as files.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - features (list): List of feature column names to include in PCA.
    - n_components (int): Number of principal components to calculate.
    - save_path (str, optional): Directory path to save the output files.
    - filename (str, optional): Filename for saving the outputs.

    Returns:
    - loadings_df (pd.DataFrame): Loadings for each feature for all principal components.
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    feature_data = data[features].dropna()
    pca_result = pca.fit_transform(feature_data)
    
    # Get the loadings (lines for original features)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Dynamically generate colors using a palette
    palette = sns.color_palette("tab10", n_colors=len(features))
    
    # Plot vectors for all component pairs
    plt.figure(figsize=(15, 12))
    plot_number = 1
    for i in range(n_components):
        for j in range(i + 1, n_components):
            plt.subplot(n_components - 1, n_components - 1, plot_number)
            
            # Add the feature vectors (lines)
            for k, feature in enumerate(features):
                plt.plot(
                    [0, loadings[k, i]], [0, loadings[k, j]],
                    color=palette[k], alpha=0.8, lw=2
                )
                plt.text(
                    loadings[k, i] * 1.1, loadings[k, j] * 1.1, 
                    feature, color=palette[k], fontsize=10
                )
            
            plt.xlabel(f'PC{i + 1}')
            plt.ylabel(f'PC{j + 1}')
            plt.title(f'PC{i + 1} vs PC{j + 1}')
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            
            plot_number += 1
    
    # Add legend
    handles = [plt.Line2D([0], [0], color=palette[i], lw=2, label=feature) 
               for i, feature in enumerate(features)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        plot_file_path = os.path.join(save_path, f"{filename}_pca_vectors.png")
        plt.savefig(plot_file_path, bbox_inches='tight')

    plt.show()
    
    # Create loadings DataFrame
    loadings_df = pd.DataFrame(loadings, columns=[f'PC{i + 1}' for i in range(n_components)], index=features)

    # Save the loadings DataFrame to CSV if save_path and filename are provided
    if save_path and filename:
        csv_file_path = os.path.join(save_path, f"{filename}_pca_loadings.csv")
        loadings_df.to_csv(csv_file_path)

    # Print PCA components (loadings)
    print("\nPCA Components (Loadings):")
    print(loadings_df)
    
    return loadings_df

##############################################################################

def plot_dendrogram(data, features, method='ward', title='Dendrogram - Hierarchical Clustering', save_path=None, filename=None):
    """
    Plot a dendrogram for hierarchical clustering and save the plot as a PNG file.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - features (list): List of feature column names to include in the dendrogram.
    - method (str): Linkage method for hierarchical clustering (default: 'ward').
    - title (str): Title of the dendrogram plot.
    - save_path (str, optional): Directory path to save the dendrogram plot.
    - filename (str, optional): Filename to save the dendrogram plot as PNG.

    Returns:
    - None
    """
    # Extract selected features from the data
    feature_data = data[features].dropna()

    # Perform hierarchical clustering
    linkage_matrix = linkage(feature_data.T, method=method)

    # Plot the dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=features, leaf_rotation=90)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Distance")

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        file_path = os.path.join(save_path, f"{filename}.png")
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()

##############################################################################
 
def perform_lda(data, features, n_components, categorical_variable, 
                title="LDA - Fisher's Discriminant", save_path=None, filename=None):
    """
    Perform Linear Discriminant Analysis (LDA) and plot histograms of the LDA results.
    Save the plot as a PNG file and the transformed data as a CSV file.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - features (list): List of feature column names to include in LDA.
    - n_components (int): Number of LDA components to calculate.
    - categorical_variable (str): Column name for the categorical variable.
    - title (str): Title of the LDA plot (default: "LDA - Fisher's Discriminant").
    - save_path (str, optional): Directory path to save the outputs.
    - filename (str, optional): Filename to save the outputs.

    Returns:
    - lda_result (pd.DataFrame): LDA-transformed data.
    - lda_model (LDA): Trained LDA model.
    """
    # Initialize LDA
    lda = LDA(n_components=n_components)
    
    # Extract features and target
    feature_data = data[features].dropna()
    target = data[categorical_variable].dropna()
    
    # Ensure target and features are aligned
    feature_data = feature_data.loc[target.index]
    
    # Perform LDA
    lda_result = lda.fit_transform(feature_data, target)
    
    # Create LDA results DataFrame
    lda_columns = [f'LD{i + 1}' for i in range(n_components)]
    lda_df = pd.DataFrame(lda_result, columns=lda_columns)
    lda_df[categorical_variable] = target.reset_index(drop=True)

    # Save LDA results to CSV if save_path and filename are provided
    if save_path and filename:
        csv_file_path = os.path.join(save_path, f"{filename}_lda_results.csv")
        lda_df.to_csv(csv_file_path, index=False)

    # Plot histograms for each category
    categories = sorted(target.unique())
    palette = sns.color_palette("tab10", n_colors=len(categories))
    
    plt.figure(figsize=(10, 6))
    for i, category in enumerate(categories):
        subset = lda_df[lda_df[categorical_variable] == category]
        plt.hist(subset[lda_columns[0]], alpha=0.7, label=f'Category {category}', color=palette[i])
    
    plt.title(title)
    plt.xlabel("LDA Component")
    plt.ylabel("Frequency")
    plt.legend()

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        plot_file_path = os.path.join(save_path, f"{filename}_lda_plot.png")
        plt.savefig(plot_file_path, bbox_inches='tight')

    plt.show()
    
    return lda_df, lda

##############################################################################

def calculate_gini(data, features, categorical_variable, criterion='gini', max_depth=3, 
                   title="Feature Importance - Gini Index", save_path=None, filename=None):
    """
    Calculate feature importance using a Decision Tree and plot the Gini index.
    Save the plot as a PNG file and the feature importance table as a CSV file.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - features (list): List of feature column names to include in the calculation.
    - categorical_variable (str): Column name for the target variable.
    - criterion (str): Splitting criterion for the Decision Tree (default: 'gini').
    - max_depth (int): Maximum depth of the Decision Tree (default: 3).
    - title (str): Title for the feature importance plot (default: "Feature Importance - Gini Index").
    - save_path (str, optional): Directory path to save the outputs.
    - filename (str, optional): Filename for saving the outputs.

    Returns:
    - feature_importances_df (pd.DataFrame): DataFrame containing features and their importance scores.
    """
    # Extract features and target
    feature_data = data[features].dropna()
    target = data[categorical_variable].dropna()
    
    # Ensure target and features are aligned
    feature_data = feature_data.loc[target.index]

    # Initialize Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    
    # Fit the model
    clf.fit(feature_data, target)
    
    # Calculate feature importances
    feature_importances = pd.Series(clf.feature_importances_, index=features)
    
    # Create a DataFrame for feature importances
    feature_importances_df = feature_importances.sort_values(ascending=False).reset_index()
    feature_importances_df.columns = ['Feature', 'Importance']
    
    # Plot feature importances
    sorted_importances = feature_importances.sort_values()
    ax = sorted_importances.plot(kind='barh', color='teal', figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    
    # Add importance values to the bars
    for i, v in enumerate(sorted_importances):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9, color='black')
    
    plt.tight_layout()

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        plot_file_path = os.path.join(save_path, f"{filename}_gini_plot.png")
        plt.savefig(plot_file_path, bbox_inches='tight')

    plt.show()

    # Save feature importances to CSV if save_path and filename are provided
    if save_path and filename:
        csv_file_path = os.path.join(save_path, f"{filename}_gini_importance.csv")
        feature_importances_df.to_csv(csv_file_path, index=False)

    # Return the table
    return feature_importances_df

##############################################################################

def calculate_mutual_information(data, features, categorical_variable, 
                                 save_path=None, filename=None):
    """
    Calculate and plot mutual information scores for features with respect to a categorical target variable.
    Save the plot as a PNG file and the feature importance table as a CSV file.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - features (list): List of feature column names.
    - categorical_variable (str): Target variable (categorical).
    - save_path (str, optional): Directory path to save the outputs.
    - filename (str, optional): Filename for saving the outputs.

    Returns:
    - mi_df (pd.DataFrame): DataFrame containing features and their mutual information scores.
    """
    # Extract feature data (X) and target data (y)
    X = data[features].dropna()
    y = data[categorical_variable].dropna()

    # Ensure target and features are aligned
    X = X.loc[y.index]

    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    # Convert scores to a DataFrame
    mi_df = pd.DataFrame({"Feature": features, "Mutual Information": mi_scores})
    
    # Sort by mutual information in descending order
    mi_df.sort_values(by="Mutual Information", ascending=False, inplace=True)
    
    # Plot mutual information scores
    plt.figure(figsize=(8, 6))
    ax = plt.barh(mi_df["Feature"], mi_df["Mutual Information"], color="skyblue")
    plt.xlabel("Mutual Information")
    plt.ylabel("Features")
    plt.title("Mutual Information Scores")
    plt.gca().invert_yaxis()  # Reverse the order of features for better readability
    
    # Add values to the bars
    for i, (feature, value) in enumerate(zip(mi_df["Feature"], mi_df["Mutual Information"])):
        plt.text(value + 0.01, i, f"{value:.3f}", va='center', fontsize=9, color='black')
    
    plt.tight_layout()

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        plot_file_path = os.path.join(save_path, f"{filename}_mutual_info_plot.png")
        plt.savefig(plot_file_path, bbox_inches='tight')

    plt.show()
    
    # Save feature mutual information to CSV if save_path and filename are provided
    if save_path and filename:
        csv_file_path = os.path.join(save_path, f"{filename}_mutual_info.csv")
        mi_df.to_csv(csv_file_path, index=False)

    return mi_df

##############################################################################

def optimal_clusters(data, features, save_path=None, filename=None):
    """
    Determine the optimal number of clusters using the Elbow Method and Silhouette Score,
    and save the combined plot as a PNG file.

    Parameters:
    - data (pd.DataFrame): Dataset containing the variables.
    - features (list): List of feature column names.
    - save_path (str, optional): Directory path to save the plot image.
    - filename (str, optional): Filename to save the plot as PNG.

    Returns:
    - None: Displays the combined plot of Elbow Method and Silhouette Method.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Prepare feature data
    X = data[features]

    # Elbow Method: Compute WCSS for k in range 1 to 10
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Silhouette Method: Compute silhouette scores for k in range 2 to 10
    silhouette_scores = []
    cluster_range = range(2, 11)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Elbow Method
    ax1.plot(range(1, 11), wcss, marker='o', linestyle='--', color='blue')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('WCSS')
    ax1.set_title('Elbow Method for Optimal k')
    for i, value in enumerate(wcss):
        ax1.text(i + 1, value, f"{value:.2f}", fontsize=9, va='bottom', ha='center')

    # Plot Silhouette Method
    ax2.plot(cluster_range, silhouette_scores, marker='o', linestyle='-', color='green')
    ax2.set_xticks(cluster_range)
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Method for Optimal Clusters')
    for i, value in enumerate(silhouette_scores):
        ax2.text(cluster_range[i], value, f"{value:.2f}", fontsize=9, va='bottom', ha='center')

    # Final adjustments
    plt.tight_layout()

    # Save the plot if save_path and filename are provided
    if save_path and filename:
        plot_file_path = os.path.join(save_path, f"{filename}_optimal_clusters.png")
        plt.savefig(plot_file_path, bbox_inches='tight')

    plt.show()

##############################################################################    
