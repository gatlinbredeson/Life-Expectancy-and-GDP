import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def get_label(col_name):
    """
    Function to retrieve the more descriptive label for a given column. Otherwise uses the column name.
    :param col_name: The string name of the column for which to retrieve the label.
    :return: The label for the column name.
    """
    try:
        lab = cols_labels[col_name]
    except KeyError:
        lab = col_name.capitalize()
    return lab


def multi_plot(dataframe, graph_type, var_x, *vars_y, uniform_axes=None, sub_plots=True):
    """
    Uses the graph_variables function to create a figure of subplots graphing any number of variables for each country.
    :param dataframe: the DataFrame wherein the data exists.
    :param graph_type: The type of graph to use, either 'plot' or 'scatter' for the purposes of this analysis.
    :param var_x: The variable to graph on the x-axis for each country.
    :param vars_y: The variables to graph on the y-axis for each country.
    :param uniform_axes: A list to set the range of all axes [x_min, x_max, y_min, y_max]
    :param sub_plots: Option to graph all data on one plot rather than individual subplots.
    :return: None
    """
    fig = plt.figure(figsize=(14, 10))

    # Iterate over each country
    for i, country in enumerate(countries_unique):
        country_frame = dataframe[(dataframe['country'] == country)]

        if sub_plots:
            fig.add_subplot(2, 3, i+1)

        x_lab = get_label(var_x)
        labels = []

        # Plot the variables for the country
        for var in vars_y:
            # Get the correct label for each variable
            y_lab = get_label(var)
            y_orig = y_lab
            labels.append(y_lab)

            if not sub_plots:
                y_lab = country

            # plot with the type of graph indicated, otherwise raise ValueError
            if graph_type == 'plot':
                plt.plot(country_frame[var_x], country_frame[var], label=y_lab)
            elif graph_type == 'scatter':
                plt.scatter(country_frame[var_x], country_frame[var], label=y_lab)
            else:
                raise ValueError

            # Set the plot labels
            plt.xlabel(x_lab)
            plt.xticks(rotation=55)
            plt.ylabel(y_orig)
            if len(vars_y) > 1:
                plt.legend()

        # Set the plot titles
        labels_expanded = ', '.join(labels)
        if sub_plots:
            plt.title(country)

        # If indicated, set axis limits
        if uniform_axes:
            plt.axis(uniform_axes)

    # If a single graph, implement a legend
    if not sub_plots:
        plt.legend()

    # Set figure spacing and title
    plt.subplots_adjust(hspace=0.35, wspace=0.35)
    fig.suptitle(f'{var_x.capitalize()} vs. {labels_expanded}')


# Get the data into a DataFrame
df = pd.read_csv('all_data.csv')

# Conform the column labels for consistency. "leaby" was "Life Expectancy at Birth (years)"
df.columns = ['country', 'year', 'leaby', 'gdp']

# Convert 'United States of America' to 'USA' for brevity
df = df.replace('United States of America', 'USA')

# A list of unique countries will be helpful to have.
countries_unique = df.country.unique()

# A dictionary of column names and their more descriptive labels
cols_labels = {
    'gdp': 'GDP (USD)',
    'gdp_normalized': 'GDP (Normalized)',
    'leaby': 'Life Expectancy (Years)',
    'leaby_normalized': 'Life Expectancy (Normalized)',
    'year': 'Years'
}

# Since the GDP values vary so widely, normalize gdp per country.
df['gdp_normalized'] = df.groupby('country').gdp.transform(lambda x: (x - x.mean()) / x.std())

# Let's do it with leaby too
df['leaby_normalized'] = df.groupby('country').leaby.transform(lambda x: (x - x.mean()) / x.std())

# Plot each country's GDP over time
multi_plot(df, 'plot', 'year', 'gdp')

# Plot each individual country's life expectancy over time to get a better sense of each's shape
multi_plot(df, 'plot', 'year', 'leaby')

# Plot all countries' life expectancy over time, to compare each country's results
multi_plot(df, 'plot', 'year', 'leaby', uniform_axes=[1999, 2016, 40, 85], sub_plots=False)

# Plot GDP and LEABY side by side on a normalized scale for each country to see if they follow one another
multi_plot(df, 'plot', 'year', 'gdp_normalized', 'leaby_normalized')

# Plot a scatter plot of GDP (Normalized) vs Life Expectancy to see the shape of the data
# Since the GDP and LEABY lines followed each other, we expect these to be positive linear
multi_plot(df, 'scatter', 'gdp_normalized', 'leaby')

# Plot GDP vs Life Expectancy fo all data, to see if there's a trend on a more general level
multi_plot(df, 'scatter', 'gdp', 'leaby', sub_plots=False)

plt.show()

# Use Linear Regression to see how strongly GDP affects Life Expectancy in each country.
# Since the GDP varies so widely, we'll normalize the GDP data in our model constructor.
for cntry in countries_unique:
    subframe = df[(df['country'] == cntry)]
    train, test = train_test_split(subframe, test_size=0.2, random_state=1000)
    lgr = LinearRegression()
    train_gdp = np.array(train.gdp_normalized).reshape(-1, 1)
    test_gdp = np.array(test.gdp_normalized).reshape(-1, 1)
    lgr.fit(train_gdp, train.leaby)
    R2 = lgr.score(test_gdp, test.leaby)
    coefs = lgr.coef_
    print(f'Country: {cntry} -- R2: %.5f -- coef: %.5f' % (R2, coefs))
