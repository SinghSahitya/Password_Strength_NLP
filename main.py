import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import string

con = sqlite3.connect('password_data.sqlite')
df = pd.read_sql_query("SELECT * FROM Users", con)
df.drop(['index'], axis=1, inplace=True)

def special_char(row):
    for char in row:
        if char in string.punctuation:
            return 1
        else:
            pass

# print(df[df['password'].apply(special_char)==1])
        
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
