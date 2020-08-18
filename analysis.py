import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import six


class DataLoader:
    """This class will be used for loading the data and making a data-frame wherever we need one."""

    def __init__(self, path):
        self.path = path

    def load_csv(self):
        df = pd.read_csv(self.path)
        return df

class DataAnalysis(DataLoader):
    """This class contains all the Data Analysis functions and this class is a child class of DataLoader because
     we need to use the functionality of the DataLoader class."""

    def revenue_over_time(self):
        """This method calculates the revenue which has been earned by the company over a period of time.
        The time has been divided on a monthly basis for the whole dataset"""

        data = super(DataAnalysis, self).load_csv()

        data['Conv_Date'] = pd.to_datetime(data['Conv_Date'])
        data['Conv_Date'] = pd.to_datetime(data['Conv_Date'], format='%Y00%m').apply(lambda x: x.strftime('%Y-%m'))
        sorted_data = data.sort_values(by=['Conv_Date'])
        sorted_data.groupby('Conv_Date')['Revenue'].sum().plot()
        plt.title('Revenue Over Time')
        plt.ylabel('Revenue', fontsize=16)
        plt.xlabel('Year-Month', fontsize=16)
        plt.show()

    def transactions_over_time(self):
        """This method calculates the number of transactions performed by the customers over a period of time. The time
        has been divided on a monthly basis for the whole dataset"""

        data = super(DataAnalysis, self).load_csv()

        data['Transactions'] = data['Conv_ID'].notnull()*1
        data['Conv_Date'] = pd.to_datetime(data['Conv_Date'])
        data['Conv_Date'] = pd.to_datetime(data['Conv_Date'], format='%Y00%m').apply(lambda x: x.strftime('%Y-%m'))
        sorted_data = data.sort_values(by=['Conv_Date'])
        sorted_data.groupby('Conv_Date')['Transactions'].sum().plot()
        plt.title('Transactions Over Time')
        plt.ylabel('Transactions', fontsize=16)
        plt.xlabel('Year-Month', fontsize=16)
        plt.show()

    def customer_types(self):
        """This method calculates the number of different types of customers. The customers have been divided in
        three categories. This method returns three values: 1. Total Customers Over the Period of the whole year for
        which the data is available. 2. The Unique Customers Over the Period of the whole year. 3. The Returning
        Customers Over the Period the whole year. """

        data = super(DataAnalysis, self).load_csv()

        total_customers = data['User_ID'].count()
        unique_customers = data['User_ID'].nunique()
        returning_customers = (data['User_ID'].duplicated(keep='first')*1).sum()

        return total_customers, unique_customers, returning_customers


    def customers_over_time(self):
        """This method analyses customers coming over a certain period of time and they have been calculated and
        plotted. The time has again been divided w.r.t months. The output is the plot of Total and Unique/New customers
        which have made a purchase over a certain month."""

        data = super(DataAnalysis, self).load_csv()

        data['Customers'] = data['User_ID'].notnull()*1
        data['Conv_Date'] = pd.to_datetime(data['Conv_Date'])
        data['Conv_Date'] = pd.to_datetime(data['Conv_Date'], format='%Y00%m').apply(lambda x: x.strftime('%Y-%m'))
        sorted_data = data.sort_values(by=['Conv_Date'])
        sorted_data.groupby('Conv_Date')['Customers'].sum().plot()

        data.drop_duplicates(subset="User_ID", keep=False, inplace=True)
        data['Unique_Customers'] = data['User_ID'].notnull() * 1
        sorted_data = data.sort_values(by=['Conv_Date'])
        sorted_data.groupby('Conv_Date')['Unique_Customers'].sum().plot()

        plt.legend(["Total Customers", "New Customers"])
        plt.title('Customers Over Time')
        plt.ylabel('Customers', fontsize=16)
        plt.xlabel('Year-Month', fontsize=16)
        plt.show()

    def customers_analysis(self):
        """In this method, the analysis of the customers has been provided. Again there are three type of customers
        which have been chosen for this analysis:
            1. Total Customers Over the Period of the whole year for which the data is available.
            2. The Unique Customers Over the Period of the whole year.
            3. The Returning Customers Over the Period the whole year.
        The output is a plot showing the difference between the number of different types of customers and also the
        values for different types of customers have been logged including the percentage for the returning
        customers"""

        total_customers, unique_customers, returning_customers = self.customer_types()

        print("Total number of Customers: {}".format(total_customers))

        print("Total number of Unique Customers: {}".format(unique_customers))

        print("The returning customers are: {}".format(returning_customers))

        print("The percentage of returning customers are: {}".format(returning_customers * 100 / total_customers))

        customers = [total_customers, unique_customers, returning_customers]
        axis = ['Total Customers', 'Unique Customers', 'Returning Customers']
        plt.bar(axis, customers)
        plt.title('Different Customers Analysis')
        plt.ylabel('Total Number', fontsize=16)
        plt.xlabel('Customer Type', fontsize=16)
        plt.show()

    def generate_table(self, data, col_width=5.0, row_height=0.625, font_size=14,
                         header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                         bbox=[0, 0, 1, 1], header_columns=0,
                         ax=None, **kwargs):
        """This method makes a table from the dataframe which we feed to it."""

        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        return ax

    def channel_analysis(self):
        """This method is analyses the most influential channels for each user. These channels can be
        thought of as marketing channels. The basic goal is to analyse which kind of channel each user gets influenced
        by and then later on a plot shows the overall most influential channels as well. Some very useful insights can
        found by the final graph showing the overall most successful channels."""

        data = super(DataAnalysis, self).load_csv()

        # making a new data frame from other table
        data_2 = pd.read_csv(r'./table_B_attribution.csv')
        idx = data_2.groupby(['Conv_ID'])['IHC_Conv'].transform(max) == data_2['IHC_Conv']

        merged = pd.merge(data, data_2[idx], on=['Conv_ID'], how='inner')

        idx_2 = merged.groupby(['User_ID'])['IHC_Conv'].transform(max) == merged['IHC_Conv']

        # plotting influential channels wrt revenue generation
        merged[idx_2].groupby(['Channel'])['Revenue'].sum().plot.bar()
        plt.title('Revenue By Channels')
        plt.xlabel('Different Channels', fontsize=16)
        plt.ylabel('Revenue Generated', fontsize=16)
        plt.show()

        # dropping irrelevant rows for this particular analysis
        final_df = merged[idx_2].drop(['Conv_ID', 'Revenue', 'Conv_Date'],axis=1)

        # plotting influential channels wrt most successful channel
        final_df.Channel.value_counts().plot.bar()
        plt.title('Overall Influential Channels')
        plt.xlabel('Different Channels', fontsize=16)
        plt.ylabel('Influence Count', fontsize=16)
        plt.show()

        # printing the most influential channels wrt different customers
        print('The influential Channels for Different Users are: \n')
        print(final_df)

        # making a table from 10 entries of the dataframe giving us the influential channels for users
        self.generate_table(final_df.head(10), header_columns=0, col_width=5.0)

        plt.show()

    def monthly_cohort_analysis(self):
        """This method does cohort analysis. The metric which I have chosen is the retention of users as the time goes
        on. It can be interesting for the company to have this analysis done so that they can analyse how the users are
        behaving with time and improve on their marketing strategies so that users can be retained over-time. This
        analysis starts from each user after their first purchase from the company."""

        data = super(DataAnalysis, self).load_csv()

        data['Conv_Date'] = pd.to_datetime(data['Conv_Date'])
        data['OrderPeriod'] = pd.to_datetime(data['Conv_Date'], format='%Y00%m').apply(lambda x: x.strftime('%Y-%m'))

        # determining cohort-group of user (based on their first order)
        data.set_index('User_ID', inplace=True)
        data['CohortGroup'] = data.groupby(level=0)['Conv_Date'].min().apply(lambda x: x.strftime('%Y-%m'))
        data.reset_index(inplace=True)

        grouped = data.groupby(['CohortGroup', 'OrderPeriod'])

        # count the unique users, orders, and total revenue per Group + Period
        cohorts = grouped.agg({'User_ID': pd.Series.nunique,
                               'Conv_ID': pd.Series.nunique,
                               'Revenue': np.sum})

        # make the column names more meaningful
        cohorts.rename(columns={'User_ID': 'TotalUsers',
                                'Conv_ID': 'TotalOrders'}, inplace=True)

        cohorts['CohortPeriod'] = np.arange(len(cohorts)) + 1

        # reindex the DataFrame
        cohorts.reset_index(inplace=True)
        cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)

        # create a Series holding the total size of each CohortGroup
        cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()

        user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)

        user_retention[['2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03']].plot(figsize=(12, 8))
        plt.title('Cohorts: User Retention')
        plt.xticks(np.arange(1, 91.1, 1))
        plt.xlim(1,91)
        plt.ylabel('% of Cohort Purchasing')

        sns.set(style='white')
        plt.figure(figsize=(12, 8))
        plt.title('Cohorts: User Retention')
        sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), cmap='coolwarm', annot=False, fmt='.0%')
        plt.show()

if __name__ == "__main__":

    conversion_path = r'./table_A_conversions.csv'

    conversions = DataAnalysis(conversion_path)

    conversions.customer_types()
    conversions.revenue_over_time()
    conversions.transactions_over_time()
    conversions.customers_over_time()
    conversions.customers_analysis()
    conversions.channel_analysis()
    conversions.monthly_cohort_analysis()










    
    
    
    
    
