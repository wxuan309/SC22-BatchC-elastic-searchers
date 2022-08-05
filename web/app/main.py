# import requirements needed
from flask import Flask, render_template
from utils import get_base_url
import pandas as pd
import pickle
from flask import request

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12347
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

def encode(values):
    df = pd.DataFrame(values).T
    df.columns = ['Make', 'Model', 'Vehicle Class', 'Fuel Type', 'Fuel Consumption Comb (mpg)']
    print(df)
    df = df.astype({'Fuel Consumption Comb (mpg)':"float"})
    col_category = ['Make', 'Model', 'Vehicle Class', 'Fuel Type']
    for col in col_category:
        df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_')], axis=1)
    df_all = pd.read_csv('CO2 Emissions_Canada.csv')
    df_all.replace(to_replace= 'Z', value= 'Premium gasoline', inplace = True)
    df_all.replace(to_replace= 'X', value= 'Gasoline' , inplace= True)
    df_all.replace(to_replace= 'D', value= 'Diesel' , inplace= True)
    df_all.replace(to_replace= 'E', value= 'Ethanol' , inplace= True)
    df_all.replace(to_replace= 'N', value= 'Natural gas' , inplace= True)
    df_all.drop(['Transmission', 'Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 
                 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)'], axis = 1, inplace = True)
    for col in col_category:
        df_all = pd.concat([df_all.drop(col, axis=1), pd.get_dummies(df_all[col], prefix=col, prefix_sep='_')], axis=1)
    all_cols = df_all.columns
    for i in all_cols:
        if i not in df:
            df[i] = 0
    df = df[all_cols]
    print(df)
    return df

# set up the routes and logic for the webserver
@app.route(f'{base_url}', methods = ["GET","POST"])
def home():
    if request.method == "POST":
        values = [i for i in request.form.values()]
        test = encode(values)
        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(test)
        print(result)
        if result[0] < 30:
            prediction = 'Your predicted CO2 emission is ' + str(result[0]) + '.' + ''
        elif result[0] < 50:
            prediction = 'Your predicted CO2 emission is ' + str(result[0]) + '.' + ''
        else:
            prediction = 'Your predicted CO2 emission is ' + str(result[0]) + '.' + ''
        return render_template('index.html', prediction = prediction)

    return render_template('index.html')


if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc5.ai-camp.dev'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
