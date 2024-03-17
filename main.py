import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.metrics import confusion_matrix ,  accuracy_score , classification_report
from warnings import filterwarnings
filterwarnings("ignore")


con = sqlite3.connect('password_data.sqlite')
df = pd.read_sql_query("SELECT * FROM Users", con)
df.drop(['index'], axis=1, inplace=True)

def special_char(row):
    for char in row:
        if char in string.punctuation:
            return 1
        else:
            pass

         
# feature engineering    
df['length'] = df['password'].str.len()

def freq_lower(row):
    return len([char for char in row if char.islower()])/len(row)

def freq_upper(row):
    return len([char for char in row if char.isupper()])/len(row)

def freq_digit(row):
    return len([char for char in row if char.isdigit()])/len(row)


def freq_specialcase(row):
    special_chars = []
    for char in row:
        if not char.isalpha() and not char.isdigit():
            special_chars.append(char)
    return len(special_chars)/len(row)


df['lowercase_freq'] = np.round(df['password'].apply(freq_lower) , 3)

df['uppercase_freq'] = np.round(df['password'].apply(freq_upper) , 3)

df['digit_freq'] = np.round(df['password'].apply(freq_digit) , 3)

df['specialchar_freq'] = np.round(df['password'].apply(freq_specialcase), 3)

cols = ['length', 'lowercase_freq', 'uppercase_freq', 'digit_freq', 'specialchar_freq']

for col in cols:
    print(col)
    print(df[[col, 'strength']].groupby(['strength']).agg(["min", "max", "mean", "median"]))
    print('\n')

#visual representation
    
fig , ((ax1 , ax2) , (ax3 , ax4) , (ax5,ax6)) = plt.subplots(3 , 2 , figsize=(15,7))

sns.boxplot(x="strength" , y='length' , hue="strength" , ax=ax1 , data=df)
sns.boxplot(x="strength" , y='lowercase_freq' , hue="strength" , ax=ax2, data=df)
sns.boxplot(x="strength" , y='uppercase_freq' , hue="strength" , ax=ax3, data=df)
sns.boxplot(x="strength" , y='digit_freq' , hue="strength" , ax=ax4, data=df)
sns.boxplot(x="strength" , y='specialchar_freq' , hue="strength" , ax=ax5, data=df)

plt.subplots_adjust(hspace=0.6)

# feature importance
def get_dist(data , feature):
    
    plt.figure(figsize=(10,8))
    plt.subplot(1,2,1)
    
    sns.violinplot(x='strength' , y=feature , data=data )
    
    plt.subplot(1,2,2)
    
    sns.distplot(data[data['strength']==0][feature] , color="red" , label="0" , hist=False)
    sns.distplot(data[data['strength']==1][feature], color="blue", label="1", hist=False)
    sns.distplot(data[data['strength']==2][feature], color="orange", label="2", hist=False)
    plt.legend()
    plt.show()

get_dist(df , "length")
get_dist(df , 'lowercase_freq')
get_dist(df , 'uppercase_freq')
get_dist(df , 'digit_freq')
get_dist(df , 'specialchar_freq')

# applying TF_IDF on data
dataframe = df.sample(frac=1) ### shuffling randomly for robustness of ML moodel 
x = list(dataframe["password"])
vectorizer = TfidfVectorizer(analyzer="char")
X = vectorizer.fit_transform(x)

df2 = pd.DataFrame(X.toarray() , columns=vectorizer.get_feature_names_out())
df2["length"] = dataframe['length']
df2["lowercase_freq"] = dataframe['lowercase_freq']
y = dataframe["strength"]

# applying machine learning algorithm
X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.20)
clf = LogisticRegression(multi_class="multinomial")
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
Counter(y_pred)

def predict(): # prediction on user input
    password = input("Enter a password : ")
    sample_array = np.array([password])
    sample_matrix = vectorizer.transform(sample_array)
    
    length_pass = len(password)
    length_normalised_lowercase = len([char for char in password if char.islower()])/len(password)
    
    new_matrix2 = np.append(sample_matrix.toarray() , (length_pass , length_normalised_lowercase)).reshape(1,101)
    result = clf.predict(new_matrix2)
    
    if result == 0 :
        return "Password is weak"
    elif result == 1 :
        return "Password is normal"
    else:
        return "password is strong"


print(classification_report(y_test , y_pred))   # model evaluation