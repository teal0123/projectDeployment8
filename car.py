import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import os
st.set_page_config(page_title="CAR PRICE",page_icon=":car:",layout="wide")
st.title(" 	:moneybag:  car price")
st.markdown("<style>div.block-containers{padding-top:1rem;}<style>",unsafe_allow_html=True)
f1=st.file_uploader(":file-folder:uploadfile",type=("csv","xls","xlxs"))
if f1 is not None:
    filename=f1.name
    st.write(filename)
    df=pd.read_csv(filename,encoding="iso-8859-1")
else:
    os.chdir(r"C:\Users\HP\Downloads\Armenian market car")
    df=pd.read_csv("Armenian Market Car Prices.csv") 
st.sidebar.header("Choose your filter: ")
import streamlit as st
import pandas as pd
# Assuming df is your dataframe
df1 = pd.DataFrame() # Load your data
# Create filters in the sidebar
region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())
year = st.sidebar.multiselect("Pick your Year", df["Year"].unique())
car_name = st.sidebar.multiselect("Pick your Car Name", df["Car Name"].unique())
fuel_type=st.sidebar.multiselect("pick fuel type",df["FuelType"].unique())
# Apply filters
df2 = df.copy()
if region:
    
    df2 =df2 [df2["Region"].isin(region)]
if year:
    df2 = df2[df2["Year"].isin(year)]

if car_name:
    df2 = df2[df2["Car Name"].isin(car_name)]
if fuel_type:
    df2=df2[df2["FuelType"].isin(fuel_type)]    

# Display the filtered dataframe
st.write(df2)
# Split 'Car Name' column into 'make' and 'model' columns

def main():
    # Load or create your DataFrame
    # For demonstration, we'll use the sample DataFrame created above
    st.write("## Original DataFrame")
    st.write(df)
    # Find null values
    null_values = df.isnull().sum()

    if null_values.sum() == 0:
        st.write("No null values detected in the DataFrame.") 
    else:
        st.write("Null values detected in the following columns:")
        st.write(null_values[null_values > 0])


# Creating a DataFrame
# Create a new DataFrame showing data types of each column
dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Types'])
dtypes_df
df[['make', 'model']] = df['Car Name'].str.split(' ', n=1, expand=True)
# Display the modified DataFrame
print(df.head())
fuel_type=df["FuelType"].value_counts()
fuel_type
# Plotting the pie chart

# Create the pie chart
fig, ax = plt.subplots()
ax.pie(fuel_type, labels=fuel_type.index, colors=["purple", "orange", "green", "pink"], autopct="%1.1f%%", explode=(0.1, 0, 0, 0))
ax.set_title('Fuel Type Distribution')
# Display the chart in Streamlit
st.title("Fuel Type Distribution")
st.pyplot(fig)

# Calculate the 10 most frequent car names
df_most_10_frequent_car_name = df['Car Name'].value_counts().head(10)

# Display the result in Streamlit
st.title('Top 10 Most Frequent Car Names in data')
st.write(df_most_10_frequent_car_name)
# Create the bar chart
fig, ax = plt.subplots()
df_most_10_frequent_car_name.plot(kind="bar", color="olive", ax=ax)
ax.set_xlabel("Car Name")
ax.set_ylabel("Count")
ax.set_title("10 Most Frequent Car Names")

# Display the chart in Streamlit
st.title("10 Most Frequent Car Names")
st.pyplot(fig)
# Sort the dataframe by price and select only 'Car Name' and 'Price' columns
sorted_10_expensive_car = df.sort_values('Price', ascending=False).head(10)[['Car Name', 'Price']]

# Display the result in Streamlit
st.title('Top 10 Most Expensive Cars')
st.write(sorted_10_expensive_car)
# Create the bar chart
fig, ax = plt.subplots()
sorted_10_expensive_car.plot(kind="bar", x="Car Name", y="Price", color="purple", ax=ax)
ax.set_xlabel("Car Name")
ax.set_ylabel("Price")
ax.set_title("10 Most Expensive Cars")

# Display the chart in Streamlit
st.title("10 Most Expensive Cars")
st.pyplot(fig)
# Sort the DataFrame by 'Price' and select only 'Car Name' and 'Price' columns
sorted_10_cheapest_car = df.sort_values('Price').head(10)[['Car Name', 'Price']]

# Display the result in Streamlit
st.title('Top 10 Cheapest Cars')
st.write(sorted_10_cheapest_car)
# Create the bar chart
fig, ax = plt.subplots()
sorted_10_cheapest_car.plot(kind="bar", x="Car Name", y="Price", color="olive", ax=ax)
ax.set_xlabel("Car Name")
ax.set_ylabel("Price")
ax.set_title("10 Cheapest Cars")

# Display the chart in Streamlit
st.title("10 Cheapest Cars")
st.pyplot(fig)
# Calculate the highest region that makes cars
highest_region_makes_car = df['Region'].value_counts().head(10)

# Create the bar chart
fig, ax = plt.subplots()
highest_region_makes_car.plot(kind="bar", color="coral", ax=ax)
ax.set_xlabel("Region")
ax.set_ylabel("Count")
ax.set_title("The Highest Region which Makes Cars")

# Display the chart in Streamlit
st.title("The Highest Region which car maker")
st.pyplot(fig)
top_5_makes_with_most_models = df.groupby('make')['model'].nunique().sort_values(ascending=False).head(5)

# Create the bar chart
fig, ax = plt.subplots()
top_5_makes_with_most_models.plot(kind="bar", color="olive", ax=ax)
ax.set_xlabel("Make")
ax.set_ylabel("Count of Models")
ax.set_title("Top 5 Car Makes with Most Models")

# Display the chart in Streamlit
st.title("Top 5 Car Makes with Most Models")
st.pyplot(fig)
# Calculate the 10 least frequent car names
least_frequent_cars = df['Car Name'].value_counts().tail(10).index.sort_values(ascending=True)
least_frequent_cars_df = df[df['Car Name'].isin(least_frequent_cars)][['Car Name', 'Price']].sort_values(by='Price', ascending=True)

# Create the bar chart
fig, ax = plt.subplots()
least_frequent_cars_df.plot(kind="bar", x="Car Name", y="Price", color="purple", ax=ax)
ax.set_xlabel("Car Name")
ax.set_ylabel("Price")
ax.set_title("10 Least Frequent Cars")

# Display the chart in Streamlit
st.title("10 Least Frequent Cars")
st.pyplot(fig)
# Get the top 10 most expensive cars for each year
yearly_top10 = df2.groupby('Year').apply(lambda x: x.nlargest(10, 'Price')[['Car Name', 'Price']])

# Display the results in Streamlit
st.write("Top 10 Most Expensive Cars by Year")

for year, top10 in yearly_top10.groupby(level=0):
    st.write(f"Top 10 Most Expensive Cars in {year}:")
    st.dataframe(top10.reset_index(drop=True))
    st.write("="*30)
 

