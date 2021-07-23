import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    :params:
    messages_filepath: Path of the files containing messages
    categories_filepath: Path of the files containing categories

    :return:
    df: dataframe with messages and categories concatenated together
    """

    # importing the messages file
    messages = pd.read_csv("./data/disaster_messages.csv")

    # importing the categories file
    categories = pd.read_csv("./data/disaster_categories.csv")

    # merging both the datasets imported above
    df = pd.merge(messages, categories, on="id")

    # separating the categories by ';'
    categories = df["categories"].str.split(";",expand = True)

    # getting the column names
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x.split("-")[0])

    # setting the column names in categories df
    categories.columns = category_colnames

    # converting the categories df to a sparse matrix
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split("-")[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # replacing the original categories column in initial df with the sparse matrix
    df.drop(["categories"], axis="columns", inplace=True)
    df = pd.concat([df, categories], axis="columns")

    return df


def clean_data(df):
    """
    :param:
    df: dataframe with messages and categories concatenated together

    :return:
    df: clean dataframe
    """

    # child_alone has zeroes, so it could be dropped
    df.drop(['child_alone'], axis='columns', inplace=True)

    # related 2 has very few values. It could be merged with the majority class 1
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    # removing duplicates
    if len(df[df.duplicated()]):
        df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    This method saves the dataframe to a sql database present in the file provided as an argument

    :param:
    df: clean dataframe
    database_filename: database file (used with create_engine)
    """

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
