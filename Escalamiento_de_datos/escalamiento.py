from scipy.io import arff
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

def read_data():

    list_data = []

    # En caso de que el archivo sea tipo arff
    # (descomenta el codigo quitando las comillas simples o el simbolo de #)
    '''
    data1 = arff.loadarff('birds.arff')
    df1 = pd.DataFrame(data1[0])
    print(df1.head())

    data2 = arff.loadarff('flags.arff')
    df2 = pd.DataFrame(data2[0])
    print(df2.head())

    data3 = arff.loadarff('yelp.arff')
    df3 = pd.DataFrame(data3[0])
    print(df3.head(3))
    '''
    # En caso de que el archivo sea tipo csv

    df1 = pd.read_csv('data1.csv')
    print(df1.head())

    df2 = pd.read_csv('data2.csv')
    print(df2.head())

    df3 = pd.read_csv('data3.csv')
    print(df3.head())

    list_data.append(df1)
    list_data.append(df2)
    list_data.append(df3)

    return list_data
    

def escalamiento(x, y):
    # splitting data train - test
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            random_state=42)
    
    print("--------StandardScaler--------")
    standard_sc = StandardScaler()
    sc = standard_sc.fit_transform(x_train)
    sc1 = standard_sc.transform(x_test)
    print("Mean")
    print(standard_sc.mean_)
    print("Min")
    print(sc.min())
    print("Max")
    print(sc.min())

    print("--------MinMaxScaler--------")
    standard_sc = MinMaxScaler()
    sc = standard_sc.fit_transform(x_train)
    sc1 = standard_sc.transform(x_test)
    print("Data Max")
    print(standard_sc.data_max_)
    print("Min")
    print(sc.min())
    print("Max")
    print(sc.min())

    print("--------RobustScaler--------")
    standard_sc = RobustScaler()
    sc = standard_sc.fit_transform(x_train)
    sc1 = standard_sc.transform(x_test)
    print("Min")
    print(sc.min())
    print("Max")
    print(sc.min())

    print("--------QuantileTransformer--------")
    standard_sc = QuantileTransformer(n_quantiles=10, random_state=0)
    sc = standard_sc.fit_transform(x_train)
    sc1 = standard_sc.transform(x_test)
    print("Min")
    print(sc.min())
    print("Max")
    print(sc.min())


def prep(list_df):
    df = list_df[0]
    # y columna del target (filtrar la columan de la clase)
    y = df.filter(["Golden Crowned Kinglet", "Warbling Vireo", "Common Nighthawk"], axis=1)
    # x columna de los datos (eliminar la columan de la clase)
    x = df.drop(["Golden Crowned Kinglet", "Warbling Vireo", "Common Nighthawk"], axis=1)
    #escalamiento(x, y)

    df = list_df[1]
    # y columna del target (filtrar la columan de la clase)
    y = df.filter(["red", "green", "blue", "yellow", "white", "black"], axis=1)
    # x columna de los datos (eliminar la columan de la clase)
    x = df.drop(["red", "green", "blue", "yellow", "white", "black"], axis=1)
    #escalamiento(x, y)

    df = list_df[2]
    # y columna del target (filtrar la columan de la clase)
    y = df.filter(["IsFoodGood", "IsServiceGood", "IsAmbianceGood", "IsDealsGood", "IsPriceGood", "IsRatingBad", "IsRatingModerate", "IsRatingGood"], axis=1)
    # x columna de los datos (eliminar la columan de la clase)
    x = df.drop(["IsFoodGood", "IsServiceGood", "IsAmbianceGood", "IsDealsGood", "IsPriceGood", "IsRatingBad", "IsRatingModerate", "IsRatingGood"], axis=1)
    escalamiento(x, y)


# Main
if __name__ == '__main__':
    list_df = read_data()
    prep(list_df)
