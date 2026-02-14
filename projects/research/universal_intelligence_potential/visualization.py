def plot_heatmap(data: pd.DataFrame):
    plt.figure(figsize=(14, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Parameters and Intelligence')
    plt.savefig('correlation_matrix.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def plot_pairplot(data: pd.DataFrame):
    sns.pairplot(data, hue='Intelligence', palette='viridis')
    plt.savefig('pairplot.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def plot_jointplot(data: pd.DataFrame):
    sns.jointplot(data=data, x='H_X', y='Intelligence', kind='hex', cmap='Blues')
    plt.savefig('jointplot.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def plot_histograms(data: pd.DataFrame):
    for column in data.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], bins=20, kde=True, color='magenta')
        plt.title(f'Distribution of {column}')
        plt.savefig(f'{column}_distribution.png')
        plt.show(block=False)
        plt.pause(5)
        plt.close()

def plot_boxplots(data: pd.DataFrame):
    for column in data.columns[:-1]:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Intelligence', y=column, data=data)
        plt.title(f'Intelligence vs {column}')
        plt.savefig(f'Intelligence_vs_{column}.png')
        plt.show(block=False)
        plt.pause(5)
        plt.close()

def plot_violinplot(data: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=data, x='Error Detection Rate', y='Correction Capability', scale='width', inner='quartile')
    plt.title('Error Detection Rate vs Correction Capability')
    plt.savefig('Error_Detection_vs_Correction_Capability_violin.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def plot_scatter_matrix(data: pd.DataFrame):
    plt.figure(figsize=(20, 15))
    pd.plotting.scatter_matrix(data, alpha=0.8, figsize=(20, 15), diagonal='kde')
    plt.savefig('scatter_matrix.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def plot_ridge_plot(data: pd.DataFrame):
    fig, axes = joypy.joyplot(data, by='Intelligence', figsize=(12, 8), colormap=plt.cm.viridis, alpha=0.8)
    plt.title('Ridge Plot of Parameters Grouped by Intelligence')
    plt.savefig('ridge_plot.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def plot_3d_scatter(data: pd.DataFrame):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['H_X'], data['P'], data['Intelligence'], c='r', marker='o')
    ax.set_xlabel('H_X')
    ax.set_ylabel('P')
    ax.set_zlabel('Intelligence')
    plt.title('3D Scatter Plot of H_X, P and Intelligence')
    plt.savefig('3D_scatter_H_X_P_Intelligence.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def visualize_results(results: pd.DataFrame):
    """
    Visualize the results using various plots.
    
    Parameters:
        results (pd.DataFrame): The results DataFrame.
    """
    numeric_results = results.drop(columns=['Probabilities'])
    print(results.to_string(index=False))

    plot_heatmap(numeric_results)
    plot_pairplot(numeric_results)
    plot_jointplot(numeric_results)
    plot_histograms(numeric_results)
    plot_boxplots(numeric_results)
    plot_violinplot(numeric_results)
    plot_scatter_matrix(numeric_results)
    plot_ridge_plot(numeric_results)
    plot_3d_scatter(numeric_results)
