#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement

# ### Business context
# 
# Employee Promotion means the ascension of an employee to higher ranks, this aspect of the job is what drives employees the most. The ultimate reward for dedication and loyalty towards an organization and the HR team plays an important role in handling all these promotion tasks based on ratings and other attributes available.
# 
# The HR team in JMD company stored data on the promotion cycle last year, which consists of details of all the employees in the company working last year and also if they got promoted or not, but every time this process gets delayed due to so many details available for each employee - it gets difficult to compare and decide.
# 
# 
# ### Objective
# 
# For the upcoming appraisal cycle, the HR team wants to utilize the stored data and leverage machine learning to make a model that will predict if a person is eligible for promotion or not. You, as a data scientist at JMD company, need to come up with the best possible model that will help the HR team to predict if a person is eligible for promotion or not.
# 
# 
# ### Data Description
# 
# - employee_id: Unique ID for the employee
# - department: Department of employee
# - region: Region of employment (unordered)
# - education: Education Level
# - gender: Gender of Employee
# - recruitment_channel: Channel of recruitment for employee
# - no_ of_ trainings: no of other training completed in the previous year on soft skills, technical skills, etc.
# - age: Age of Employee
# - previous_ year_ rating: Employee Rating for the previous year
# - length_ of_ service: Length of service in years
# - awards_ won: if awards won during the previous year then 1 else 0
# - avg_ training_ score: Average score in current training evaluations
# - is_promoted: (Target) Recommended for promotion

# ## Importing necessary libraries

# In[1]:


# Installing the libraries with the specified version.
# uncomment and run the following line if Google Colab is being used
# !pip install scikit-learn==1.2.2 seaborn==0.13.1 matplotlib==3.7.1 numpy==1.25.2 pandas==1.5.3 imbalanced-learn==0.10.1 xgboost==2.0.3 -q --user


# In[2]:


# Installing the libraries with the specified version.
# uncomment and run the following lines if Jupyter Notebook is being used
# !pip install scikit-learn==1.2.2 seaborn==0.13.1 matplotlib==3.7.1 numpy==1.25.2 pandas==1.5.3 imbalanced-learn==0.10.1 xgboost==2.0.3 -q --user
# !pip install --upgrade -q threadpoolctl


# In[3]:


# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# Libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# To tune model, get different metric scores, and split data
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# To be used for data scaling and one hot encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# To oversample and undersample data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# To do hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# To impute missing values
from sklearn.impute import SimpleImputer

# To define maximum number of columns to be displayed in a dataframe
pd.set_option("display.max_columns", None)

# To supress scientific notations for a dataframe
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# To help with model building
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier

# To suppress scientific notations
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# To supress warnings
import warnings

warnings.filterwarnings("ignore")


# ## Loading the dataset

# In[4]:


promotion = pd.read_csv("employee_promotion.csv")


# ## Data Overview

# The initial steps to get an overview of any dataset is to:
# - observe the first few rows of the dataset, to check whether the dataset has been loaded properly or not
# - get information about the number of rows and columns in the dataset
# - find out the data types of the columns to ensure that data is stored in the preferred format and the value of each property is as expected.
# - check the statistical summary of the dataset to get an overview of the numerical columns of the data

# ### Checking the shape of the dataset

# In[5]:


# Checking the number of rows and columns in the training data
promotion.shape ##  Complete the code to view dimensions of the train data


# In[6]:


# let's create a copy of the data
data = promotion.copy()


# ### Displaying the first few rows of the dataset

# In[7]:


# let's view the first 5 rows of the data
data.head() ##  Complete the code to view top 5 rows of the data


# In[8]:


# let's view the last 5 rows of the data
data.tail() ##  Complete the code to view last 5 rows of the data


# ### Checking the data types of the columns for the dataset

# In[9]:


# let's check the data types of the columns in the dataset
data.info()


# ### Checking for duplicate values

# In[10]:


# let's check for duplicate values in the data
data.duplicated() ##  Complete the code to check duplicate entries in the data


# ### Checking for missing values

# In[11]:


# let's check for missing values in the data
data.isnull().sum() ##  Complete the code to check missing entries in the train data


# ### Statistical summary of the dataset

# In[12]:


# let's view the statistical summary of the numerical columns in the data
data.describe() ##  Complete the code to print the statitical summary of the train data


# In[13]:


# let's view the statistical summary of the numerical columns in the data
data.describe(include= ["object"]).T


# **Let's check the number of unique values in each column**

# In[14]:


data.nunique()


# In[15]:


for i in data.describe(include=["object"]).columns:
    print("Unique values in", i, "are :")
    print(data[i].value_counts())
    print("*" * 50)


# In[16]:


# ID column consists of uniques ID for clients and hence will not add value to the modeling
data.drop(columns="employee_id", inplace=True)


# In[17]:


data["is_promoted"].value_counts(1)


# ## Exploratory Data Analysis

# #### The below functions need to be defined to carry out the Exploratory Data Analysis.

# In[18]:


# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[19]:


# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# In[20]:


# function to plot stacked bar chart

def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# In[21]:


### Function to plot distributions

def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


# ### Univariate analysis

# #### Observations on No. of Trainings

# In[22]:


histogram_boxplot(data, "no_of_trainings")


# **Let's see the distribution of age of employee**

# #### Observations on Age

# In[23]:


histogram_boxplot(data, "age")  ## Complete the code to create histogram_boxplot for 'age'


# #### Observations on Length of Service

# In[24]:


histogram_boxplot(data, "length_of_service")  ## Complete the code to create histogram_boxplot for 'length_of_service'


# **Let's see the distribution of average training score of employee**

# #### Observations on Average Training Score

# In[25]:


histogram_boxplot(data, "avg_training_score")  ## Complete the code to create histogram_boxplot for 'avg_training_score'


# #### Observations on Department

# In[26]:


labeled_barplot(data, "department")


# #### Observations on Education

# In[27]:


labeled_barplot(data, "education")
#labeled_barplot('_______') ## Complete the code to create labeled_barplot for 'education'


# #### Observations on Gender

# In[28]:


labeled_barplot(data, "gender") ## Complete the code to create labeled_barplot for 'gender'


# #### Observations on Recruitment Channel

# In[29]:


labeled_barplot(data, "recruitment_channel") ## Complete the code to create labeled_barplot for 'recruitment_channel'


# #### Observations on Previous Year Rating

# In[30]:


labeled_barplot(data, "previous_year_rating") ## Complete the code to create labeled_barplot for 'previous_year_rating'


# #### Observations on Awards Won

# In[31]:


labeled_barplot(data, "awards_won") ## Complete the code to create labeled_barplot for 'awards_won'


# #### Observations on Region

# In[32]:


labeled_barplot(data, "region") ## Complete the code to create labeled_barplot for 'region'


# #### Observations on target variable

# In[33]:


labeled_barplot(data, "is_promoted") ## Complete the code to create labeled_barplot for 'is_promoted'


# ### Bivariate Analysis

# In[86]:


sns.pairplot(data, hue="is_promoted")


# #### Target variable vs Age

# In[35]:


distribution_plot_wrt_target(data, "age", "is_promoted")


# **Let's see the change in length of service (length_of_service) vary by the employee's promotion status (is_promoted)?**

# #### Target variable vs Length of Service

# In[36]:


distribution_plot_wrt_target(data, "length_of_service", "is_promoted") ## Complete the code to create distribution_plot for length_of_service vs is_promoted


# #### Target variable vs Average Training Score

# In[37]:


distribution_plot_wrt_target(data, "avg_training_score", "is_promoted") ## Complete the code to create distribution_plot for avg_training_score vs is_promoted


# #### Target variable vs Department

# In[38]:


stacked_barplot(data, "department", "is_promoted")


# #### Target variable vs Region

# In[39]:


stacked_barplot(data, "region", "is_promoted")


# #### Target variable vs Education

# In[40]:


stacked_barplot(data,"education", "is_promoted") ## Complete the code to create distribution_plot for education vs is_promoted


# #### Target variable vs Gender

# In[41]:


stacked_barplot(data,"gender", "is_promoted") ## Complete the code to create distribution_plot for gender vs is_promoted


# #### Target variable vs Recruitment Channel

# In[42]:


stacked_barplot(data,"recruitment_channel", "is_promoted") ## Complete the code to create distribution_plot for recruitment_channel vs is_promoted


# **Let's see the previous rating(previous_year_rating) vary by the employee's promotion status (is_promoted)**

# #### Target variable vs Previous Year Rating

# In[43]:


stacked_barplot(data,"previous_year_rating", "is_promoted") ## Complete the code to create distribution_plot for previous_year_rating vs is_promoted


# #### Target variable vs Awards Won

# In[44]:


stacked_barplot(data,"awards_won", "is_promoted") ## Complete the code to create distribution_plot for awards_won vs is_promoted


# In[45]:


sns.boxplot(data=data, x="awards_won", y="avg_training_score")


# **Let's see the attributes that have a strong correlation with each other**

# ### Correlation Heatmap

# In[46]:


plt.figure(figsize=(15, 7))
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# ## Data Preprocessing

# In[47]:


data1 = data.copy()


# ### Train-Test Split

# In[48]:


X = data1.drop(["is_promoted"], axis=1)
y = data1["is_promoted"]


# In[49]:


# Splitting data into training and validation set:

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size= 0.2, random_state= 42) ## Complete the code to split the data into train test in the ratio 80:20

X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size= 0.25, random_state= 42) ## Complete the code to split the data into train test in the ratio 75:25

print(X_train.shape, X_val.shape, X_test.shape)


# ### Missing value imputation

# In[50]:


# Defining the imputers for numerical and categorical variables
imputer_mode = SimpleImputer(strategy="most_frequent")
imputer_median = SimpleImputer(strategy="median")


# In[51]:


# Fit and transform the train data
X_train[["education"]] = imputer_mode.fit_transform(X_train[["education"]])

# Transform the validation data
X_val[["education"]]  =  imputer_mode.transform(X_val[["education"]])  ## Complete the code to impute missing values in X_val

# Transform the test data
X_test[["education"]] =  imputer_mode.transform(X_test[["education"]]) ## Complete the code to impute missing values in X_test


# In[52]:


# Fit and transform the train data
X_train[["previous_year_rating", "avg_training_score"]] = imputer_median.fit_transform(
    X_train[["previous_year_rating", "avg_training_score"]]
)

# Transform the validation data
X_val[["previous_year_rating", "avg_training_score"]]  =  imputer_median.transform(
    X_val[["previous_year_rating", "avg_training_score"]]) ## Complete the code to impute missing values in X_val

# Transform the test data
X_test[["previous_year_rating", "avg_training_score"]] = imputer_median.transform(
    X_test[["previous_year_rating", "avg_training_score"]]
)
 ## Complete the code to impute missing values in X_test


# In[53]:


# Checking that no column has missing values in train, validation and test sets
print(X_train.isna().sum())
print("-" * 30)
print(X_val.isna().sum())
print("-" * 30)
print(X_test.isna().sum())


# ### Encoding categorical variables

# In[54]:


X_train = pd.get_dummies(X_train, drop_first=True)
X_val = pd.get_dummies(X_val, drop_first=True)  ## Complete the code to impute missing values in X_val
X_test = pd.get_dummies(X_test, drop_first=True)  ## Complete the code to impute missing values in X_val
print(X_train.shape, X_val.shape, X_test.shape)
print(X_train.shape, X_test.shape)


# ## Building the model

# ### Model evaluation criterion

# **Model can make wrong predictions as:**
# 
# - Predicting an employee should get promoted when he/she should not get promoted
# - Predicting an employee should not get promoted when he/she should get promoted
# 
# **Which case is more important?**
# 
# - Both cases are important here as not promoting a deserving employee might lead to less productivity and the company might lose a good employee which affects the company's growth. Further, giving promotion to a non-deserving employee would lead to loss of monetary resources and giving such employee higher responsibility might again affect the company's growth.
# 
# **How to reduce this loss i.e need to reduce False Negatives as well as False Positives?**
# 
# - Bank would want `F1-score` to be maximized, as both classes are important here. Hence, the focus should be on increasing the F1-score rather than focusing on just one metric i.e. Recall or Precision.

# **First, let's create two functions to calculate different metrics and confusion matrix, so that we don't have to use the same code repeatedly for each model.**

# In[55]:


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred, average="macro")  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
        },
        index=[0],
    )

    return df_perf


# In[56]:


def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ### Model Building - Original Data

# In[57]:


models = []  # Empty list to store all the models

# Appending models into the list
models.append(("Bagging", BaggingClassifier(random_state=1)))
models.append(("Random forest", RandomForestClassifier(random_state=1)))
models.append(("AdaBoost", AdaBoostClassifier(random_state=1)))
models.append(("Gradient Boosting", GradientBoostingClassifier(random_state=1)))
models.append(("XGBoost", XGBClassifier(random_state=1, eval_metric='logloss'))) ## Complete the code to append remaining 4 models in the list models

results1 = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models

scorer= 'f1_macro'
# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Cost:" "\n")

for name, model in models:
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train, y=y_train, scoring=scorer, cv=kfold
    )
    results1.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean()))

print("\n" "Validation Performance:" "\n")

for name, model in models:
    model.fit(X_train, y_train)
    scores = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores))


# In[58]:


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results1)
ax.set_xticklabels(names)

plt.show()


# ### Model Building - Oversampled Data

# In[59]:


print("Before Oversampling, counts of label 'Yes': {}".format(sum(y_train == 1)))
print("Before Oversampling, counts of label 'No': {} \n".format(sum(y_train == 0)))

sm = SMOTE(
    sampling_strategy=1, k_neighbors=5, random_state=1
)  # Synthetic Minority Over Sampling Technique
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)


print("After Oversampling, counts of label 'Yes': {}".format(sum(y_train_over == 1)))
print("After Oversampling, counts of label 'No': {} \n".format(sum(y_train_over == 0)))


print("After Oversampling, the shape of train_X: {}".format(X_train_over.shape))
print("After Oversampling, the shape of train_y: {} \n".format(y_train_over.shape))


# In[60]:


models = []  # Empty list to store all the models

# Appending models into the list
models.append(("Bagging", BaggingClassifier(random_state=1)))
models.append(("Random forest", RandomForestClassifier(random_state=1)))
models.append(("AdaBoost", AdaBoostClassifier(random_state=1)))
models.append(("Gradient Boosting", GradientBoostingClassifier(random_state=1)))
models.append(("XGBoost", XGBClassifier(random_state=1, eval_metric='logloss'))) ## Complete the code to append remaining 4 models in the list models

results1 = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models

scorer= 'f1_macro'
# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Cost:" "\n")

for name, model in models:
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train_over, y=y_train_over, scoring=scorer, cv=kfold
    )
    results1.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean()))

print("\n" "Validation Performance:" "\n")

for name, model in models:
    model.fit(X_train_over, y_train_over)
    scores = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores))
## Complete the code to build models on oversampled data
## Note - Take reference from the original models built above


# In[61]:


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results1)
ax.set_xticklabels(names)

plt.show() ## Write the code to create boxplot to check model performance on oversampled data


# ### Model Building - Undersampled Data

# In[62]:


rus = RandomUnderSampler(random_state=1)
X_train_un, y_train_un = rus.fit_resample(X_train, y_train)


# In[63]:


print("Before Under Sampling, counts of label 'Yes': {}".format(sum(y_train == 1)))
print("Before Under Sampling, counts of label 'No': {} \n".format(sum(y_train == 0)))

print("After Under Sampling, counts of label 'Yes': {}".format(sum(y_train_un == 1)))
print("After Under Sampling, counts of label 'No': {} \n".format(sum(y_train_un == 0)))

print("After Under Sampling, the shape of train_X: {}".format(X_train_un.shape))
print("After Under Sampling, the shape of train_y: {} \n".format(y_train_un.shape))


# In[64]:


models = []  # Empty list to store all the models

# Appending models into the list
models.append(("Bagging", BaggingClassifier(random_state=1)))
models.append(("Random forest", RandomForestClassifier(random_state=1)))
models.append(("AdaBoost", AdaBoostClassifier(random_state=1)))
models.append(("Gradient Boosting", GradientBoostingClassifier(random_state=1)))
models.append(("XGBoost", XGBClassifier(random_state=1, eval_metric='logloss'))) ## Complete the code to append remaining 4 models in the list models

results1 = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models

scorer= 'f1_macro'
# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Cost:" "\n")

for name, model in models:
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train_un, y=y_train_un, scoring=scorer, cv=kfold
    )
    results1.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean()))

print("\n" "Validation Performance:" "\n")

for name, model in models:
    model.fit(X_train_un, y_train_un)
    scores = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores))
## Complete the code to build models on oversampled data
## Note - Take reference from the original models built above


# In[65]:


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results1)
ax.set_xticklabels(names)

plt.show() ## Write the code to create boxplot to check model performance on oversampled data


# ### Hyperparameter Tuning

# #### **Note**
# 1. Sample parameter grid has been provided to do necessary hyperparameter tuning. One can extend/reduce the parameter grid based on execution time and system configuration to try to improve the model performance further wherever needed.      
# 2. The models chosen in this notebook are based on test runs. One can update the best models as obtained upon code execution and tune them for best performance.
# 
# 

# #### Tuning AdaBoost using Undersampled data

# In[66]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nModel = AdaBoostClassifier(random_state=1)\n\n# Parameter grid to pass in RandomSearchCV\nparam_grid = {\n    "n_estimators": np.arange(10, 110, 10),\n    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],\n    "base_estimator": [\n        DecisionTreeClassifier(max_depth=1, random_state=1),\n        DecisionTreeClassifier(max_depth=2, random_state=1),\n        DecisionTreeClassifier(max_depth=3, random_state=1),\n    ],\n}\n\n# Type of scoring used to compare parameter combinations\nscorer = \'f1_macro\'\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_jobs = -1, n_iter=50, scoring=scorer, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train_un, y_train_un) ## Complete the code to fit the model on undersampled data\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))\n')


# In[67]:


# Creating new pipeline with best parameters
tuned_adb1 = AdaBoostClassifier( random_state=1,
    n_estimators= 50, learning_rate= 1, base_estimator= DecisionTreeClassifier(max_depth=2, random_state=1)
) ## Complete the code with the best parameters obtained from tuning

tuned_adb1.fit(X_train_un, y_train_un) ## Complete the code to fit the model on undersampled data


# In[68]:


adb1_train = model_performance_classification_sklearn(tuned_adb1, X_train_un, y_train_un) ## Complete the code to check the performance on training set
adb1_train


# In[69]:


# Checking model's performance on validation set
adb1_val =  model_performance_classification_sklearn(tuned_adb1, X_val, y_val) ## Complete the code to check the performance on validation set
adb1_val


# #### Tuning AdaBoost using original data

# In[70]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nModel = AdaBoostClassifier(random_state=1)\n\n# Parameter grid to pass in RandomSearchCV\nparam_grid = {\n    "n_estimators": np.arange(10, 110, 10),\n    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],\n    "base_estimator": [\n        DecisionTreeClassifier(max_depth=1, random_state=1),\n        DecisionTreeClassifier(max_depth=2, random_state=1),\n        DecisionTreeClassifier(max_depth=3, random_state=1),\n    ],\n}\n\n# Type of scoring used to compare parameter combinations\nscorer = \'f1_macro\'\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_jobs = -1, n_iter=50, scoring=scorer, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train, y_train) ## Complete the code to fit the model on original data\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))\n')


# In[71]:


# Creating new pipeline with best parameters
tuned_adb2 = AdaBoostClassifier( random_state=1,
    n_estimators= 90, learning_rate= 1, base_estimator= DecisionTreeClassifier(max_depth=2, random_state=1)
) ## Complete the code with the best parameters obtained from tuning

tuned_adb2.fit(X_train, y_train) ## Complete the code to fit the model on original data


# In[72]:


adb2_train = model_performance_classification_sklearn(tuned_adb2, X_train, y_train) ## Complete the code to check the performance on training set
adb2_train


# In[73]:


# Checking model's performance on validation set
adb2_val = model_performance_classification_sklearn(tuned_adb2, X_val, y_val)  ## Complete the code to check the performance on validation set
adb2_val


# #### Tuning Gradient Boosting using undersampled data

# In[74]:


get_ipython().run_cell_magic('time', '', '\n#Creating pipeline\nModel = GradientBoostingClassifier(random_state=1)\n\n#Parameter grid to pass in RandomSearchCV\nparam_grid = {\n    "init": [AdaBoostClassifier(random_state=1),DecisionTreeClassifier(random_state=1)],\n    "n_estimators": np.arange(75,150,25),\n    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],\n    "subsample":[0.5,0.7,1],\n    "max_features":[0.5,0.7,1],\n}\n\n# Type of scoring used to compare parameter combinations\nscorer = \'f1_macro\'\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, scoring=scorer, cv=5, random_state=1, n_jobs = -1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train_un, y_train_un) ## Complete the code to fit the model on under sampled data\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))\n')


# In[75]:


# Creating new pipeline with best parameters
tuned_gbm1 = GradientBoostingClassifier(
    max_features=0.7,
    init=AdaBoostClassifier(random_state=1),
    random_state=1,
    learning_rate=0.2,
    n_estimators=125,
    subsample=0.7,
)## Complete the code with the best parameters obtained from tuning

tuned_gbm1.fit(X_train_un, y_train_un)


# In[76]:


gbm1_train = model_performance_classification_sklearn(tuned_gbm1, X_train_un, y_train_un) ## Complete the code to check the performance on oversampled train set
gbm1_train


# In[77]:


gbm1_val = model_performance_classification_sklearn(tuned_gbm1, X_val, y_val) ## Complete the code to check the performance on validation set
gbm1_val


# #### Tuning Gradient Boosting using original data

# In[78]:


get_ipython().run_cell_magic('time', '', '\n#defining model\nModel = GradientBoostingClassifier(random_state=1)\n\n#Parameter grid to pass in RandomSearchCV\nparam_grid = {\n    "init": [AdaBoostClassifier(random_state=1),DecisionTreeClassifier(random_state=1)],\n    "n_estimators": np.arange(75,150,25),\n    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],\n    "subsample":[0.5,0.7,1],\n    "max_features":[0.5,0.7,1],\n}\n\n# Type of scoring used to compare parameter combinations\nscorer = \'f1_macro\'\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, scoring=scorer, cv=5, random_state=1, n_jobs = -1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train, y_train) ## Complete the code to fit the model on original data\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))\n')


# In[79]:


# Creating new pipeline with best parameters
tuned_gbm2 = GradientBoostingClassifier(
    max_features=0.7,
    init=AdaBoostClassifier(random_state=1),
    random_state=1,
    learning_rate=0.2,
    n_estimators=125,
    subsample=0.7,
)## Complete the code with the best parameters obtained from tuning

tuned_gbm2.fit(X_train, y_train)


# In[80]:


gbm2_train = model_performance_classification_sklearn(tuned_gbm2, X_train, y_train) ## Complete the code to check the performance on original data
gbm2_train


# In[81]:


gbm2_val = model_performance_classification_sklearn(tuned_gbm2, X_val, y_val) ## Complete the code to check the performance on validation set
gbm2_val


# ## Model Comparison and Final Model Selection

# In[82]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        gbm1_train.T,
        gbm2_train.T,
        adb1_train.T,
        adb2_train.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Gradient boosting trained with Undersampled data",
    "Gradient boosting trained with Original data",
    "AdaBoost trained with Undersampled data",
    "AdaBoost trained with Original data",
]
print("Training performance comparison:")
models_train_comp_df


# In[83]:


# validation performance comparison

models_train_comp_df = pd.concat(
    [
        gbm1_val.T,
        gbm2_val.T,
        adb1_val.T,
        adb2_val.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Gradient boosting trained with Undersampled data",
    "Gradient boosting trained with Original data",
    "AdaBoost trained with Undersampled data",
    "AdaBoost trained with Original data",
]
print("Validation performance comparison:")
models_train_comp_df ## Write the code to compare the performance on validation set


# **Now we have our final model, so let's find out how our final model is performing on unseen test data.**

# In[89]:


# Let's check the performance on test set
y_test_pred = tuned_gbm1.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy}")
print("Confusion Matrix:")
print(test_confusion_matrix) ## Write the code to check the performance of best model on test data


# ### Feature Importances

# In[85]:


feature_names = X_train.columns
importances =  tuned_adb2.feature_importances_ ## Complete the code to check the feature importance of the best model
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# # Business Insights and Conclusions
# 

# -Use the improved model to enhance targeted strategies, like focusing on previously underrepresented customer segments to boost engagement and retention.
# 
# -Regularly evaluate model metrics like recall and precision to ensure it continues to perform well as new data comes in, especially for the minority class.
# 

# ***
